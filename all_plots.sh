#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")


for d in $(ls $1 | grep -E 'DLIO|DLRM')
do  
    echo $d
    python3 plot_timelines.py $1/$d $d
    python3 plot_paper.py $1/$d $d
done

