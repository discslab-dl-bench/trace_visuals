#!/bin/bash

if [ $# -lt 3 ]
then    
    echo "Usage: $0 gpu_trace.out output_dir num_gpus"
    exit -1
fi

gpu_trace=$1
output_dir=$2
num_gpus=$3


if [[ ! -d $output_dir/gpu_data ]]
then
    echo "Creating output directory $output_dir/gpu_data"
    mkdir $output_dir/gpu_data
fi

# Extract raw data only, for calculating the average more easily
# Filtering on python will extract only the gpus used for training
# $output_dir/gpu_data/gpu.all

while read -r line
do
	(awk -v num_gpus=${num_gpus} -v original_line="${line}" '{ if ($3 < num_gpus) { print original_line } }' <<< $line) >> $output_dir/gpu_data/gpu.all
done < <(grep -E "python" $gpu_trace)
