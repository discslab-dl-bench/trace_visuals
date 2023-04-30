#!/bin/bash


# Short plots
python3 plot_timelines.py paper_plot_traces/UNET_8g4b_noinstru unet3d short -po -s 
python3 plot_timelines.py paper_plot_traces/DLIO_UNET_8GPU_4b_new_ckpt dlio short -po -s 
python3 plot_timelines.py paper_plot_traces/DLIO_UNET_8GPU_4b_new_ckpt_2 dlio short -po -s 


python3 plot_timelines.py paper_plot_traces/DLRM_LATEST_8G_32kglobal_32ks dlrm short -po -s
python3 plot_timelines.py paper_plot_traces/DLIO_DLRM_8GPU_b32k_new_ckpt dlio short -po -s
python3 plot_timelines.py paper_plot_traces/DLIO_DLRM_8GPU_b32k_new_ckpt_2 dlio short -po -s

python3 plot_timelines.py paper_plot_traces/BERT_8GPU_sda bert short -po -s
python3 plot_timelines.py paper_plot_traces/BERT_8GPU_sda_2 bert short -po -s
python3 plot_timelines.py paper_plot_traces/DLIO_BERT_8GPU_6b_new_ckpt dlio short -po -s
python3 plot_timelines.py paper_plot_traces/DLIO_BERT_8GPU_6b_new_ckpt_2 dlio short -po -s


