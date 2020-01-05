#!/bin/sh

calibrate_classification_models() {
    model_list=("resnet50_v1" "resnet50_v1b" "resnet50_v1c" "resnet50_v1d" "resnet50_v1s" "resnet50_v1d_0.11" \
                "resnet18_v1" "resnet101_v1" "mobilenet1.0" "mobilenetv2_1.0" \
                "squeezenet1.0" "squeezenet1.1" "vgg16")
    # enter ./classification/imagenet
    cd ./classification/imagenet
    for net_name in ${model_list[@]}
    do
        model_file="./model/${net_name}-quantized-naive-symbol.json"
        if [ -f ${model_file} ];then
            echo "${model_file} has existed. Skipping calibration..."
            continue
        fi
        python verify_pretrained.py --model=${net_name} --batch-size=1 --calibration --batch-size 4
    done

    # inception-v3
    python verify_pretrained.py --model=inceptionv3 --batch-size=1 --calibration --batch-size 4 --input-size 299
    # reture ./scripts
    cd ../..
}

calibrate_detection_models() {
    # backbone_list=("vgg16_atrous" "mobilenet1.0" "resnet50_v1")

    # cd ./detection/ssd
    # for net_name in ${backbone_list[@]}
    # do
    #     model_file="./model/ssd_512_${net_name}_voc-quantized-naive-symbol.json"
    #     if [ -f ${model_file} ];then
    #         echo "${model_file} has existed. Skipping calibration..."
    #         continue
    #     fi
    #     python eval_ssd.py --network=${net_name} --data-shape=512 --batch-size=4 --calibration
    # done

    backbone_list=("darknet53" "mobilenet1.0")
    dataset_list=("voc" "coco")
    cd ./detection/yolo
    for net_name in ${backbone_list[@]}
    do
    	for dataset in ${dataset_list[@]}
    	do
    		model_file="./model/yolo3_${net_name}_${dataset}-quantized-naive-symbol.json"
    		if [ -f ${model_file} ];then
	            echo "${model_file} has existed. Skipping calibration..."
	            continue
        	fi
        	python eval_yolo.py --network=${net_name} --gpus='' --calibration --calib-mode=naive --dataset=${dataset}
    	done
    done
    cd ../..
}


calibrate_segmentation_models() {
    model_list=("fcn_resnet101_coco" "psp_resnet101_coco" "deeplab_resnet101_coco" "fcn_resnet101_voc" \
                "psp_resnet101_voc" "deeplab_resnet101_voc")
    # enter ./segmentation
    cd ./segmentation
    for net_name in ${model_list[@]}
    do
        model_file="./model/${net_name}-quantized-naive-symbol.json"
        if [ -f ${model_file} ];then
            echo "${model_file} has existed. Skipping calibration..."
            continue
        fi
        python test.py --eval --calibration --batch-size=1 --num-calib-batches=1 --model=${net_name} --backbone=resnet101 --calib-mode naive  
    done
    # reture ./scripts
    cd ..
}

calibrate_pose_estimation_models() {
    model_list=("simple_pose_resnet18_v1b" "simple_pose_resnet50_v1b" "simple_pose_resnet50_v1d" \
                "simple_pose_resnet101_v1b" "simple_pose_resnet101_v1d")
    # enter ./segmentation
    cd ./pose/simple_pose
    for net_name in ${model_list[@]}
    do
        model_file="./model/${net_name}-quantized-naive-symbol.json"
        if [ -f ${model_file} ];then
            echo "${model_file} has existed. Skipping calibration..."
            continue
        fi
        python validate.py --calibration --model ${net_name} --batch-size 1 --num-calib-batches 1 --num-joints 17 --calib-mode naive
    done
    # reture ./scripts
    cd ../..
}

calibrate_action_recognition_models() {
    model_list=("vgg16_ucf101")
    # enter ./segmentation
    cd ./action-recognition
    for net_name in ${model_list[@]}
    do
        model_file="./model/${net_name}-quantized-naive-symbol.json"
        if [ -f ${model_file} ];then
            echo "${model_file} has existed. Skipping calibration..."
            continue
        fi
        python test_recognizer.py --batch-size 1 --model vgg16_ucf101 --new-height 256 --new-width 340 --use-pretrained --calibration --calib-mode naive
    done

    model_list2=("inceptionv3_ucf101")
    for net_name in ${model_list2[@]}
    do
        python test_recognizer.py --model inceptionv3_ucf101 --new-height 340 --new-width 450 --input-size 299 --use-pretrained --batch-size 1 --calibration --calib-mode naive
    done
    # reture ./scripts
    cd ../
}

benchmark_classification_symbolic_models() {
    model_list=("resnet50_v1" "resnet50_v1b" "resnet50_v1c" "resnet50_v1d" "resnet50_v1s" "resnet50_v1d_0.11" \
                "resnet18_v1" "resnet101_v1" "mobilenet1.0" "mobilenetv2_1.0" \
                "squeezenet1.0" "squeezenet1.1" "vgg16")
    # enter ./classification/imagenet
    cd ./classification/imagenet
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 ${net_name} model..."
        taskset -c 0-$2 python verify_pretrained_symbolic.py --model=resnet50_v1 --deploy --model-prefix=./model/${net_name} --benchmark --batch-size=$1
        echo ">>> Benchmarking INT8 ${net_name} model..."
        taskset -c 0-$2 python verify_pretrained_symbolic.py --model=resnet50_v1 --deploy --model-prefix=./model/${net_name}-quantized-naive --benchmark --batch-size=$1
    done

    # inception-v3
    echo ">>> Benchmarking FP32 inceptionv3 model..."
    taskset -c 0-$2 python verify_pretrained_symbolic.py --model=inceptionv3 --deploy --model-prefix=./model/inceptionv3 --input-size 299 --benchmark --batch-size=$1
    echo ">>> Benchmarking INT8 inceptionv3 model..."
    taskset -c 0-$2 python verify_pretrained_symbolic.py --model=inceptionv3 --deploy --model-prefix=./model/inceptionv3-quantized-naive --input-size 299 --benchmark --batch-size=$1
    # reture ./scripts
    cd ../..
}

benchmark_classification_gluon_models() {
    model_list=("resnet50_v1" "resnet50_v1b" "resnet50_v1c" "resnet50_v1d" "resnet50_v1s" "resnet50_v1d_0.11" \
                "resnet18_v1" "resnet101_v1" "mobilenet1.0" "mobilenetv2_1.0" \
                "squeezenet1.0" "squeezenet1.1" "vgg16")
    # enter ./classification/imagenet
    cd ./classification/imagenet
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 ${net_name} model..."
        taskset -c 0-$2 python verify_pretrained.py --model=resnet50_v1 --deploy --model-prefix=./model/${net_name} --benchmark --batch-size=$1
        echo ">>> Benchmarking INT8 ${net_name} model..."
        taskset -c 0-$2 python verify_pretrained.py --model=resnet50_v1 --deploy --model-prefix=./model/${net_name}-quantized-naive --benchmark --batch-size=$1
    done

    # inception-v3
    echo ">>> Benchmarking FP32 inceptionv3 model..."
    taskset -c 0-$2 python verify_pretrained.py --model=inceptionv3 --deploy --model-prefix=./model/inceptionv3 --input-size 299 --benchmark --batch-size=$1
    echo ">>> Benchmarking INT8 inceptionv3 model..."
    taskset -c 0-$2 python verify_pretrained.py --model=inceptionv3 --deploy --model-prefix=./model/inceptionv3-quantized-naive --input-size 299 --benchmark --batch-size=$1
    # reture ./scripts
    cd ../..
}

benchmark_detection_symbolic_models() {
    backbone_list=("vgg16_atrous" "mobilenet1.0" "resnet50_v1")
    # enter ./classification/imagenet
    cd ./detection/ssd
    for net_name in ${backbone_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python eval_ssd_symbolic.py --network=${net_name} --data-shape=512 --batch-size=$1 --deploy --model-prefix=./model/ssd_512_${net_name}_voc --benchmark
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python eval_ssd_symbolic.py --network=${net_name} --data-shape=512 --batch-size=$1 --deploy --model-prefix=./model/ssd_512_${net_name}_voc-quantized-naive --benchmark
    done
    cd ../..
}

benchmark_detection_gluon_models() {
    # backbone_list=("vgg16_atrous" "mobilenet1.0" "resnet50_v1")
    # cd ./detection/ssd
    # for net_name in ${backbone_list[@]}
    # do
    #     echo ">>> Benchmarking FP32 model..."
    #     taskset -c 0-$2 python eval_ssd.py --network=${net_name} --data-shape=512 --batch-size=$1 --deploy --model-prefix=./model/ssd_512_${net_name}_voc --benchmark
    #     echo ">>> Benchmarking INT8 model..."
    #     taskset -c 0-$2 python eval_ssd.py --network=${net_name} --data-shape=512 --batch-size=$1 --deploy --model-prefix=./model/ssd_512_${net_name}_voc-quantized-naive --benchmark
    # done
    
    backbone_list=("darknet53" "mobilenet1.0")
    dataset_list=("voc" "coco")
    cd ./detection/yolo
    for net_name in ${backbone_list[@]}
    do
    	for dataset in ${dataset_list[@]}
    	do
    		echo ">>> Benchmarking FP32 model..."
	        # taskset -c 0-$2 python eval_yolo.py --network=${net_name} --batch-size=$1 --deploy --model-prefix=./model/yolo3_${net_name}_${dataset} --benchmark
	        taskset -c 0-$2 python eval_yolo.py --network=${net_name} --gpus='' --dataset=${dataset} --batch-size=$1 --benchmark

	        echo ">>> Benchmarking INT8 model..."
	        taskset -c 0-$2 python eval_yolo.py --network=${net_name} --gpus='' --batch-size=$1 --deploy --model-prefix=./model/yolo3_${net_name}_${dataset}-quantized-naive --benchmark
    	done
    done
    cd ../..
}

benchmark_segmentation_symbolic_models() {
    model_list=("fcn_resnet101_coco" "psp_resnet101_coco" "deeplab_resnet101_coco" "fcn_resnet101_voc" \
                "psp_resnet101_voc" "deeplab_resnet101_voc")
    # enter ./segmentation
    cd ./segmentation
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python test_symbolic.py --model-prefix=./model/${net_name} --benchmark --batch-size $1
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python test_symbolic.py --model-prefix=./model/${net_name}-quantized-naive --benchmark --batch-size $1
    done
    # reture ./scripts
    cd ..
}

benchmark_segmentation_gluon_models() {
    model_list=("fcn_resnet101_coco" "psp_resnet101_coco" "deeplab_resnet101_coco" "fcn_resnet101_voc" \
                "psp_resnet101_voc" "deeplab_resnet101_voc")
    # enter ./segmentation
    cd ./segmentation
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python test.py --deploy --model-prefix=./model/${net_name} --benchmark --batch-size $1
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python test.py --deploy --model-prefix=./model/${net_name}-quantized-naive --benchmark --batch-size $1
    done
    # reture ./scripts
    cd ..
}

benchmark_pose_estimation_symbolic_models() {
    model_list=("simple_pose_resnet18_v1b" "simple_pose_resnet50_v1b" "simple_pose_resnet50_v1d" \
                "simple_pose_resnet101_v1b" "simple_pose_resnet101_v1d")
    # enter ./segmentation
    cd ./pose/simple_pose
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python validate_symbolic.py --deploy --model-prefix ./model/${net_name} --num-joints 17 --model simple --batch-size $1 --benchmark
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python validate_symbolic.py --deploy --model-prefix ./model/${net_name}-quantized-naive --num-joints 17 --model simple --batch-size $1 --benchmark
    done
    # reture ./scripts
    cd ../..
}

benchmark_pose_estimation_gluon_models() {
    model_list=("simple_pose_resnet18_v1b" "simple_pose_resnet50_v1b" "simple_pose_resnet50_v1d" \
                "simple_pose_resnet101_v1b" "simple_pose_resnet101_v1d")
    # enter ./segmentation
    cd ./pose/simple_pose
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python validate.py --deploy --model-prefix ./model/${net_name} --num-joints 17 --model simple --batch-size $1 --benchmark
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python validate.py --deploy --model-prefix ./model/${net_name}-quantized-naive --num-joints 17 --model simple --batch-size $1 --benchmark
    done
    # reture ./scripts
    cd ../..
}

benchmark_action_recognition_symbolic_models() {
    model_list=("vgg16_ucf101")
    # enter ./action-recognition
    cd ./action-recognition
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python test_recognizer_symbolic.py --model ucf --num-segments 3 --model-prefix ./model/${net_name} --deploy --new-height 256 --new-width 340 --deploy --batch-size $1 --benchmark
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python test_recognizer_symbolic.py --model ucf --num-segments 3 --model-prefix ./model/${net_name}-quantized-naive --deploy --new-height 256 --new-width 340 --deploy --batch-size $1 --benchmark
    done

    model_list=("inceptionv3_ucf101")
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-27 python test_recognizer_symbolic.py --model ucf --num-segments 3 --model-prefix ./model/${net_name} --input-size 299 --new-height 340 --new-width 450 --deploy --batch-size $1 --benchmark 
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-27 python test_recognizer_symbolic.py --model ucf --num-segments 3 --model-prefix ./model/${net_name}-quantized-naive --input-size 299 --new-height 340 --new-width 450 --deploy --batch-size $1 --benchmark
    done
    # reture ./scripts
    cd ../
}

benchmark_action_recognition_gluon_models() {
    model_list=("vgg16_ucf101")
    # enter ./action-recognition
    cd ./action-recognition
    for net_name in ${model_list[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        echo "taskset -c 0-$2 python test_recognizer.py --model ucf --num-segments 3 --model-prefix ./model/${net_name} --deploy --new-height 256 --new-width 340 --deploy --batch-size $1 --benchmark"
        taskset -c 0-$2 python test_recognizer.py --model ucf --num-segments 3 --model-prefix ./model/${net_name} --deploy --new-height 256 --new-width 340 --deploy --batch-size $1 --benchmark
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python test_recognizer.py --model ucf --num-segments 3 --model-prefix ./model/${net_name}-quantized-naive --deploy --new-height 256 --new-width 340 --deploy --batch-size $1 --benchmark
    done

    model_list2=("inceptionv3_ucf101")
    for net_name in ${model_list2[@]}
    do
        echo ">>> Benchmarking FP32 model..."
        taskset -c 0-$2 python test_recognizer.py --model ucf --num-segments 3 --model-prefix ./model/${net_name} --input-size 299 --new-height 340 --new-width 450 --deploy --batch-size $1 --benchmark 
        echo ">>> Benchmarking INT8 model..."
        taskset -c 0-$2 python test_recognizer.py --model ucf --num-segments 3 --model-prefix ./model/${net_name}-quantized-naive --input-size 299 --new-height 340 --new-width 450 --deploy --batch-size $1 --benchmark
    done
    # reture ./scripts
    cd ../
}

export cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print $4}')
export OMP_NUM_THREADS=$cpus
export BIND_MAX_CORE_ID=$[$OMP_NUM_THREADS-1]
echo "Binding processes to 0-${BIND_MAX_CORE_ID}"
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export PYTHONPATH=~/github/incubator-mxnet/python:~/github/gluon-cv

# echo "INFO: Starting calibrating FP32 classification models..."
# calibrate_classification_models > ./logs/calib_imagenet_models.log 2>&1 
# echo "INFO: Starting benchmarking classification models..."
# benchmark_classification_symbolic_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_imagenet_symbolic_models.log 2>&1 
# benchmark_classification_gluon_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_imagenet_gluon_models.log 2>&1 
# rm $(pwd)/classification/imagenet/model/*

echo "INFO: Starting calibrating FP32 detection models..."
calibrate_detection_models > calib_detection_models.log 2>&1
echo "INFO: Starting benchmarking detection models..."
# benchmark_detection_symbolic_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_detection_symbolic_models.log 2>&1
benchmark_detection_gluon_models 64 ${BIND_MAX_CORE_ID} > benchmark_detection_gluon_models_bs-64.log 2>&1
grep speed benchmark_detection_gluon_models_bs-64.log | awk '{ speed = $(NF-1) }; END { print "Inference speed with batch-size 64 is " speed " images/sec"}'
benchmark_detection_gluon_models 1 ${BIND_MAX_CORE_ID} > benchmark_detection_gluon_models_bs-1.log 2>&1
grep speed benchmark_detection_gluon_models_bs-1.log | awk '{ speed = $(NF-1) }; END { print "Inference speed with batch-size 1 is " speed " images/sec"}'
# rm $(pwd)/detection/ssd/model/*

# echo "INFO: Starting calibrating FP32 segmentation models..."
# calibrate_segmentation_models > ./logs/calib_segmentation_models.log 2>&1 
# echo "INFO: Starting benchmarking segmentation models..."
# benchmark_segmentation_symbolic_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_segmentation_symbolic_models.log 2>&1
# benchmark_segmentation_gluon_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_segmentation_gluon_models.log 2>&1
# rm $(pwd)/segmentation/model/*

# echo "INFO: Starting calibrating FP32 pose estimation models..."
# calibrate_pose_estimation_models > ./logs/calib_estimation_models.log 2>&1 
# echo "INFO: Starting benchmarking pose estimation models..."
# benchmark_pose_estimation_symbolic_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_estimation_symbolic_models.log 2>&1
# benchmark_pose_estimation_gluon_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_estimation_gluon_models.log 2>&1
# rm $(pwd)/pose/simple_pose/model/*

# echo "INFO: Starting calibrating FP32 action recognition models..."
# calibrate_action_recognition_models > ./logs/calib_recognition_models.log 2>&1
# echo "INFO: Starting benchmarking action recognition models..."
# benchmark_action_recognition_symbolic_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_recognition_symbolic_models.log 2>&1
# benchmark_action_recognition_gluon_models 64 ${BIND_MAX_CORE_ID} > ./logs/benchmark_recognition_gluon_models.log 2>&1
# rm $(pwd)/action-recognition/model/*
