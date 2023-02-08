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
            experiment_name="BERT_${method}_${num_gpus}gpu_${batch_size}b_1200steps"
            python3 preprocess_traces.py data/${experiment_name} bert
            python3 timeline.py data_preprocessed/${experiment_name} bert $experiment_name
        done
    done
done
