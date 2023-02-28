#!/bin/bash

dirname=$(basename $1)

python3 preprocess_traces.py $1 $2 -o $3
python3 plot_timelines.py $3/$dirname $dirname
python3 plot_paper.py $3/$dirname $dirname
