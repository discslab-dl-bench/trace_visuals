#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")




for d in $(ls $1 | grep DLIO)
do  
    echo $d
    python3 preprocess_traces.py $1/$d dlio -o data_feb24_proc/
done

for d in $(ls $1 | grep -E '^DLRM')
do  
    echo $d
    python3 preprocess_traces.py $1/$d dlrm -o data_feb24_proc/
done

# for d in $(ls $1 | grep UNET)
# do  
#     echo $d
#     python3 preprocess_traces.py $1/$d unet3d -o data_feb24_proc/
# done
# for d in $(ls $1 | grep -E '^BERT')
# do  
#     echo $d
#     python3 preprocess_traces.py $1/$d bert -o data_feb24_proc/
# done

# for d in $(ls data | grep 32ksteps)
# do  
#     echo $d
#     python3 preprocess_traces.py data/$d dlrm
#     python3 plot_timelines.py data_processed/$d/timeline dlrm $d
# done

# for d in $(ls data | grep BERT_horovod)
# do  
#     echo $d
#     python3 preprocess_traces.py data/$d bert 
#     python3 plot_timelines.py data_processed/$d/timeline bert $d
# done