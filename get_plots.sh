#!/bin/bash


cd data
traces_dir=$(echo $1 | awk -F "/" '{print $2}') # name of the dir contains all raw trace results (no need to add data/ in front)
ta_traces_dir="ta_${traces_dir}"
# num_gpus=$2 # number of gpus used in the current experiment
# exp_name=$3 # the name you want for the experiment plots dir
py=python3


if [ $# -lt 1 ]
then
    # example: ./get_plots.sh data/trace0_exp_results
	echo "Usage: $0 <traces_dir>"
	exit 1
fi


# preprocessing traces
for trace_dir in "$traces_dir"/*; do
	trace_expname=$(basename $trace_dir)
	num_gpus=${trace_expname:0:1}
	if [[ ! $trace_expname == *"ta"* ]]; then
		echo "Start preprocessing $trace_dir..."
		./preprocess_traces.sh $trace_dir $num_gpus
	fi
	
done


# install the missing modules in the python used in bash (uncomment the next 2 lines if you are running the script for the first time)
# pip install matplotlib
# pip install IPython

# run plots generating script
cd ..
for ta_trace_dir in "data/$traces_dir"/ta_*; do
	exp_name=$(echo $ta_trace_dir | awk -F "/" '{print $NF}' | awk -F "_" '{print $2"_"$3}')
	echo $exp_name
	echo $ta_trace_dir
	${py} timeline.py $ta_trace_dir $exp_name
done