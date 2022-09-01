#!/bin/bash


if [ $# -lt 1 ]
then
    # example: ./generate_plots.sh data/trace0_exp_results
	echo "Usage: $0 <traces_dir>"
	exit 1
fi

if [ $# -eq 2 ] # consider the optional second argument
then
	mode=$2
else
	mode="all"
fi
# Kill the tmux session from a previous run if it exists
tmux kill-session -t plots 2>/dev/null
# Start a new tmux session from which we will run training
tmux new-session -d -s plots
tmux send-keys -t plots "./preprocess_plot.sh $1 ${mode}" C-m
