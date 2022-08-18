#!/bin/bash


cd data
traces_dir=$1 # name of the dir contains all raw trace results
ta_traces_dir="ta_${traces_dir}"
num_gpus=$2 # number of gpus used in the current experiment
exp_name=$3 # the name you want for the experiment plots dir
py=python3


if [ $# -lt 3 ]
then
    # example: ./get_plots.sh exp_4gpus_20220818150543 4 4gpu
	echo "Usage: $0 <traces_dir> <numgpus> <experiment_name>"
	exit 1
fi


# preprocessing traces
./preprocess_traces.sh $traces_dir $num_gpus

cd ..
# install the missing modules in the python used in bash (uncomment the next 2 lines if you are running the script for the first time)
# pip install matplotlib
# pip install IPython
${py} timeline.py data/$ta_traces_dir $exp_name