#!/bin/bash

mkdir -p data_step_breakdown/UNET3D_sleep_4/raw_data

for d in $( ls data/mar18/UNET3D_sleep_4 )
do
    echo $d
    cp data/mar18/UNET3D_sleep_4/$d/processed/unet3d.log data_step_breakdown/UNET3D_sleep_4/raw_data/$d.json
done

python3 step_breakdown.py data_step_breakdown/UNET3D_sleep_4 unet3d


exit




trace_dir="/data/lhovon/pre_march_traces"

# declare -a traces=(UNET_bugfix_runs UNET_latest_w_original UNET_instrumented_v4_w_step7 UNET_instrumented_feb27 UNET_original_1w UNET_original_1w_2)
declare -a traces=(UNET_bugfix_runs UNET_latest_w_original UNET_instrumented_v4_w_step7)


for trace in "${traces[@]}";
do 
    echo $trace
    mkdir -p data/UNET3D/$trace

    for d in $(ls $trace_dir/$trace )
    do  
        echo $d
        tar xvzf $trace_dir/$trace/$d -C data/UNET3D/$trace
    done

    for d in $(ls data/UNET3D/$trace )
    do  
        echo $d
        python3 preprocess_traces.py data/UNET3D/$trace/$d unet3d
    done

done





# python3 throughputs.py data/mar16/DLIO_UNET_formula unet3d --title "dlio"



# mkdir -p data_step_breakdown/UNET3D_load_only/raw_data



# python3 step_breakdown.py data_step_breakdown/UNET3D_load_only unet3d
