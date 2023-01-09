#!/bin/bash

# Simply makes valid JSON out of the unet3d.log files in a list of directories
# then moves the resulting files in train_times/ for further processing with training_times.py 

for d in $(ls -d */ | grep -E '^UNET_[48]GPU_batch1.*'); do ./mllog2validjson.sh ${d}unet3d.log; done

for d in $(ls -d */ | grep -E '^UNET_[48]GPU_batch1.*'); do { DIR=$(basename $d); echo $DIR; mv ${d}/unet3d.log_valid.json train_times/${DIR}.json; } done