# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
from tqdm import tqdm
import numpy as np
import time
import sys
import argparse
import logging

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset
from gluoncv.utils.viz import get_color_pallete

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Segmentation model')
    # model and dataset 
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='base network')
    parser.add_argument('--image-shape', type=int, default=480,
                        help='image shape')
    parser.add_argument('--model-prefix', type=str, default='',
                        help='load static model')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset used for validation [pascal_voc, pascal_aug, coco, ade20k]')
    parser.add_argument('--quantized', action='store_true', 
                        help='whether to use quantized model')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--benchmark', action='store_true',
                        help='using dummy data to benchmark performance')

    args = parser.parse_args()
    return args


def evaluate(net, bs, ctx, num_workers, dataset, logger):
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    testset = get_segmentation_dataset(
            dataset, split='val', mode='val', transform=input_transform)
    size = len(testset)
    test_data = gluon.data.DataLoader(
            testset, bs, last_batch='rollover', num_workers=num_workers)
    logger.info('Batch size for inference is %d' % bs)

    # warm up
    dry_run = 5
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=ctx, dtype='float32')
                for _, shape in net.data_shapes]
    batch = mx.io.DataBatch(data, [])
    for i in range(dry_run):
        outputs = model.forward(batch.data[0])
        for output in outputs:
            output.wait_to_read()

    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)
    metric.reset()
    tbar = tqdm(test_data)
    tic = time.time()
    for i, (batch, dsts) in enumerate(tbar):
        targets = mx.gluon.utils.split_and_load(dsts, ctx_list=[ctx], even_split=False)
        data = batch.as_in_context(ctx)
        net.forward(mx.io.DataBatch([data]), is_train=False)
        outputs = net.get_outputs()
        metric.update(targets, outputs)
        pixAcc, mIoU = metric.get()
        tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    speed = size / (time.time() - tic)
    logger.info('Throughput is %f img/sec' % speed)

def benchmark(net, args, input_shape, ctx, num_batches, batch_size, logger):
    if not isinstance(ctx, list):
        ctx = [ctx]

    size = num_batches * batch_size
    data = [mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx[0], dtype='float32')]
    batch = mx.io.DataBatch(data, [])

    # from mxnet import profiler
    # if args.quantized:
    #     filename = '_symbolic_profile_int8_BS{}.json'.format(str(input_shape[0]))
    # else:
    #     filename = '_symbolic_profile_fp32_BS{}.json'.format(str(input_shape[0]))
    # profiler.set_config(profile_all=True,
    #                     aggregate_stats=True,
    #                     continuous_dump=True,
    #                     filename=args.model + filename)

    dry_run = 5
    with tqdm(total=size+dry_run*bs) as pbar:
        for n in range(dry_run + num_batches):
            if n == dry_run:
                # profiler.set_state('run')
                tic = time.time()
            mod.forward(batch, is_train=False)
            for output in mod.get_outputs():
                output.wait_to_read()
            pbar.update(bs)
    #     profiler.set_state('stop')
    # profiler.dump()
    speed = size / (time.time() - tic)
    logger.info('Throughput is %f imgs/sec' % speed)


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    ctx = mx.cpu(0)
    bs = args.batch_size
    CHANNEL_COUNT = 3
    num_batches = args.num_batches
    image_shape = args.image_shape
    input_shape = (bs, CHANNEL_COUNT, image_shape, image_shape)

    if args.backbone not in ['resnet101', 'resnet50']:
        raise ValueError('Unsupported base network {} for fcn'.format(args.backbone)) 

    model_prefix = args.model + '_' + args.backbone
    if 'pascal' in args.dataset:
        model_prefix += '_voc'
    elif args.dataset == 'coco':
        model_prefix += '_coco'
    elif args.dataset == 'ade20k':
        model_prefix += '_ade'
    else:
        raise ValueError('Unimplemented dataset {} used'.format(args.dataset))

    if args.quantized:
        model_prefix += '_int8'

    # sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, 0)
    logger.info('Successfully loaded symbol from %s ' % (args.model_prefix))
    mod = mx.module.Module(sym, data_names=('data',), label_names=None, fixed_param_names=sym.list_arguments())
    mod.bind(data_shapes=[('data', input_shape)], for_training=False, grad_req=None)
    mod.set_params(arg_params, aux_params)

    if 'fcn' in args.model_prefix:
        args.model = 'fcn'
    elif 'psp' in args.model_prefix:
        args.model = 'psp'
    else:
        args.model = 'deeplab'

    if 'quantized' in args.model_prefix:
        args.quantized = True

    if not args.benchmark:
        evaluate(mod, bs, ctx, args.num_workers, args.dataset, logger)
    else:
        benchmark(mod, args, input_shape, ctx, num_batches, bs, logger)
    logger.info('Inference on {} model has been done!'.format(args.model))
