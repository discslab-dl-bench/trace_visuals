#!/bin/bash


cd data
traces_dir=$(echo $1 | awk -F "/" '{print $2}') # name of the dir contains all raw trace results (no need to add data/ in front)
ta_traces_dir="ta_${traces_dir}"
py=python3


if [ $# -lt 1 ]
then
    # example: ./get_plots.sh data/trace0_exp_results
	echo "Usage: $0 <traces_dir>"
	exit 1
fi


# remove previously preprocessed data
sudo rm -rf ${traces_dir}/ta*

# preprocessing traces
for trace_dir in "$traces_dir"/*; do
	trace_expname=$(basename $trace_dir)
	trace_expdir=$(dirname $trace_dir)
	if ! [[ $trace_expname == "ta"* ]]; then
		num_gpus=${trace_expname:0:1}
		trace_tadir="${trace_expdir}/ta_${trace_expname}"
		echo "Start preprocessing $trace_dir..."
		./preprocess_traces.sh $trace_dir $num_gpus
		# if [ ! -d "$trace_tadir" ]; then
			
		# fi
	fi
	
done


# install the missing modules in the python used in bash (uncomment the next 2 lines if you are running the script for the first time)
# pip install matplotlib
# pip install IPython

# run plots generating script
cd ..
for ta_trace_dir in "data/$traces_dir"/ta_*; do
	exp_name=$(echo $ta_trace_dir | awk -F "/" '{print $NF}' | awk -F "_" '{print $2"_"$3"_"$4}')
	echo $exp_name
	echo $ta_trace_dir
	${py} timeline.py $ta_trace_dir $exp_name
done