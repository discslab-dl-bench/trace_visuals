#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")


TRACE_DIR="/dl-bench/lhovon/tracing_tools/trace_results"


for d in $(ls data3 | grep DLRM_)
do  
    echo $d
    mkdir -p data3_proc/$d
    python3 preprocess_traces.py data3/$d dlrm -o data3_proc | tee data3_proc/${d}/${d}.log
    python3 plot_timelines.py data3_proc/$d dlrm $d
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