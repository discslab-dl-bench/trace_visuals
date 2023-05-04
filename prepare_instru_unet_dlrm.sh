#!/bin/bash

if [ ! $# -eq 2 ]
then
    echo "Usage: $0 data_dir ['unet3d', 'dlrm']"
    exit -1
fi

DATA_DIR=$1
WORKLOAD=$2
EXPERIMENT=$(basename $DATA_DIR)

mkdir -p instrumentation_data/$EXPERIMENT/raw_data

for d in $( ls $DATA_DIR )
do
    echo $d

    # Preprocess the application log
    python3 preprocess_traces.py $DATA_DIR/$d unet3d -ml

    # Copy the processed application log file to the instrumentation data folder
    cp $DATA_DIR/$d/processed/$WORKLOAD.log instrumentation_data/$EXPERIMENT/raw_data/$d.json
done

# Process the instrumentation data and generate plots
python3 proc_instru_data.py instrumentation_data/$EXPERIMENT $WORKLOAD

