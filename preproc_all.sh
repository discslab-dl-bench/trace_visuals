#!/bin/bash

declare -a num_gpus=(4 8)
declare -a batch_sizes=(6)
declare -a methods=("horovod")

for num_gpu in "${num_gpus[@]}"
do  
    for batch_size in "${batch_sizes[@]}"
    do
        for method in "${methods[@]}"
        do
            pushd data
            experiment_name="BERT_${method}_${num_gpus}gpu_${batch_size}b_1200steps"

            ./preprocess_traces.sh $experiment_name $num_gpu bert

            popd 
            /usr/bin/python3 timeline.py data/ta_${experiment_name} bert $experiment_name
        done
    done
done



# declare -a num_gpus=(2 4 6 8)

# for num_gpu in "${num_gpus[@]}";
# do  
#     for batch_size in $(seq 1 5)
#     do
#         pushd data
#         experiment_name="UNET_${num_gpu}GPU_batch${batch_size}"
#         folder_name="UNET_${num_gpu}GPU_batch${batch_size}_instrumented"

#         ./preprocess_traces.sh $folder_name $num_gpu unet3d

#         popd 
#         python3 timeline.py data/ta_${folder_name} unet3d $experiment_name
#     done
# done