#!/bin/bash

# Preprocess traces and generate timeline plots for the paper

# UNET3D
python3 preprocess_traces.py timeline_data/UNET3D_8g4b unet3d
python3 plot_timelines.py timeline_data/UNET3D_8g4b unet3d original -po -s

python3 preprocess_traces.py timeline_data/UNET3D_benchmark_8GPU_4b dlio
python3 plot_timelines.py timeline_data/UNET3D_benchmark_8GPU_4b dlio UNET3D_benchmark -po -s 

# BERT
python3 preprocess_traces.py timeline_data/BERT_8gb6 bert
python3 plot_timelines.py timeline_data/BERT_8gb6 bert original -po -s

python3 preprocess_traces.py timeline_data/BERT_benchmark_8g_b6 dlio
python3 plot_timelines.py timeline_data/BERT_benchmark_8g_b6 dlio BERT_benchmark -po -s

# DLRM
python3 preprocess_traces.py timeline_data/DLRM_8G_32kglobal dlrm
python3 plot_timelines.py timeline_data/DLRM_8G_32kglobal dlrm original -po -s

python3 preprocess_traces.py timeline_data/DLRM_benchmark_8GPU_b32k dlio
python3 plot_timelines.py timeline_data/DLRM_benchmark_8GPU_b32k dlio DLRM_benchmark -po -s

