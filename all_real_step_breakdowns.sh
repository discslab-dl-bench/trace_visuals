#!/bin/bash

python3 step_breakdown_2.py data_step_breakdown/UNET_bugfix unet3d
python3 step_breakdown_2.py data_step_breakdown/UNET_2 unet3d

python3 step_breakdown_2.py data_step_breakdown/DLRM_LARGE_2 dlrm

python3 step_breakdown_bert.py BERT_profiler