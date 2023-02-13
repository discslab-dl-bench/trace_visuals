#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")

for d in $(ls data | grep BERT_horovod)
do  
    echo $d
    python3 preprocess_traces.py data/$d bert
    # python3 plot_timelines.py data_processed/$d/timeline bert $d
done


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