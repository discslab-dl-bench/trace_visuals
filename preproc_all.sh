#!/bin/bash

declare -a num_gpus=(2 4 6 8)

for num_gpu in "${num_gpus[@]}";
do  
    for batch_size in $(seq 1 5)
    do
        pushd data
        experiment_name="UNET_${num_gpu}GPU_batch${batch_size}"
        folder_name="UNET_${num_gpu}GPU_batch${batch_size}_instrumented"

        ./preprocess_traces.sh $folder_name $num_gpu unet3d

        popd 
        python3 timeline.py data/ta_${folder_name} unet3d $experiment_name
    done
done
