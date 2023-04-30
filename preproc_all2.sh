#!/bin/bash



trace_dir="/data/lhovon/pre_march_traces"

declare -a traces=(UNET_instrumented_feb27 UNET_original_1w UNET_original_1w_2)



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




# for d in $(ls data/mar16/UNET3D_nostep7 )
# do  
#     echo $d
#     python3 preprocess_traces.py data/mar16/UNET3D_nostep7/$d unet3d
# done

# mkdir -p data_step_breakdown/UNET3D_nostep7/raw_data

# for d in $( ls data/mar16/UNET3D_nostep7 )
# do
#     echo $d
#     cp data/mar16/UNET3D_nostep7/${d}/processed/unet3d.log data_step_breakdown/UNET3D_nostep7/raw_data/${d}.json
# done

# python3 step_breakdown.py data_step_breakdown/UNET3D_nostep7 unet3d
