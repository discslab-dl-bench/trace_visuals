#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")




for d in $(ls $1 | grep -E 'UNET')
do  
    echo $d
    python3 preprocess_traces.py $1/$d unet3d -o UNET_instru_proc/
done



# for d in $(ls data | grep BERT_horovod)
# do  
#     echo $d
#     python3 preprocess_traces.py data/$d bert 
#     python3 plot_timelines.py data_processed/$d/timeline bert $d
# done