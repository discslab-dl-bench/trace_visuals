#!/bin/bash

# declare -a num_gpus=(4 8)
# declare -a batch_sizes=(6)
# declare -a methods=("horovod")


TRACE_DIR="/dl-bench/lhovon/tracing_tools/trace_results"

python3 preprocess_traces.py /dl-bench/lhovon/tracing_tools/trace_results/DLRM_TB_bin_1w dlrm
python3 plot_timelines.py data_processed/DLRM_TB_bin_1w dlrm DLRM_TB_bin_1w

python3 preprocess_traces.py /dl-bench/lhovon/tracing_tools/trace_results/UNET_30GB_generated unet3d
python3 plot_timelines.py data_processed/UNET_30GB_generated unet3d UNET_30GB_generated

exit 0


for d in $(ls $TRACE_DIR | grep UNET_200GB)
do  
    echo $d
    mkdir data2_proc/${d}
    python3 preprocess_traces.py data2/$d unet3d -o data2_proc > data2_proc/${d}/${d}.log
    python3 plot_timelines.py data_processed/$d/timeline unet3d $d
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