#!/bin/bash


preprocess(){
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

		fi
	done
}


plotting(){
	# run plots generating script
	cd ..
	for ta_trace_dir in "data/$traces_dir"/ta_*; do
		exp_name=$(echo $ta_trace_dir | awk -F "/" '{print $NF}' | awk -F "_" '{print $2"_"$3"_"$4}')
		echo $exp_name
		echo $ta_trace_dir
		${py} timeline.py $ta_trace_dir $exp_name
	done	
}

main(){
	cd data
	traces_dir=$(echo $1 | awk -F "/" '{print $2}') # name of the dir contains all raw trace results (no need to add data/ in front)
	ta_traces_dir="ta_${traces_dir}"
	py=python3


	if [ $# -lt 1 ]
	then
		# example: ./get_plots.sh data/trace0_exp_results
		echo "Usage: $0 <traces_dir> (<mode>)"
		exit 1
	fi

	if [ $# -eq 2 ] # consider the optional second argument
	then
		mode=$2
	else
		mode="all"
	fi

	if [[ "$mode" == "all" ]]
	then
		preprocess
		plotting
	elif [[ "$mode" == "plot" ]]
	then
		plotting
	
	elif [[ "$mode" == "pre" ]] # pre stands for preprocess
	then
		preprocess
	fi

}

# Call the main function with all arguments
main $@