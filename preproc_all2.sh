#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")


for d in $(ls $1 | grep -E '^DLIO' )
do  
    echo $d
    python3 preprocess_traces.py $1/$d dlio -o mar14proc/
    python3 plot_timelines.py mar14proc/$d/timeline $d
done

for d in $(ls $1 | grep -E '^UNET3D' )
do  
    echo $d
    python3 preprocess_traces.py $1/$d unet3d -o mar14proc/
    python3 plot_timelines.py mar14proc/$d/timeline $d
done

for d in $(ls $1 | grep -E '^DLRM' )
do  
    echo $d
    python3 preprocess_traces.py $1/$d dlrm -o mar14proc/
    python3 plot_timelines.py mar14proc/$d/timeline $d
done