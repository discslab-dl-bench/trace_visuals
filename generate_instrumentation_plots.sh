#!/bin/bash

# Generate the step breakdown and throughput plots

# UNET3D
python3 proc_instru_data.py \
    instrumentation_data/UNET3D_original_1 \
    instrumentation_data/UNET3D_original_2 \
    instrumentation_data/UNET3D_original_3 \
    instrumentation_data/UNET3D_original_4 \
    unet3d -o instrumentation_data/UNET3D_original_combined --breakdown --throughputs --latencies

python3 proc_instru_data.py \
    instrumentation_data/UNET3D_gen_1 \
    instrumentation_data/UNET3D_gen_2 \
    unet3d --title gen -o instrumentation_data/UNET3D_gen_combined --throughputs --latencies

python3 proc_instru_data.py \
    instrumentation_data/UNET3D_sleep_1 \
    instrumentation_data/UNET3D_sleep_2 \
    unet3d --title sleep -o instrumentation_data/UNET3D_sleep_combined --throughputs --latencies

python3 proc_instru_data_dlio.py \
    instrumentation_data/UNET3D_benchmark_1 \
    instrumentation_data/UNET3D_benchmark_2 \
    instrumentation_data/UNET3D_benchmark_3 \
    instrumentation_data/UNET3D_benchmark_4 \
    unet3d -o instrumentation_data/UNET3D_benchmark_combined --title vldb --throughputs --latencies


#BERT
python3 proc_instru_data_bert.py \
    instrumentation_data/BERT_original_1 \
    instrumentation_data/BERT_original_2 \
    instrumentation_data/BERT_original_3 \
    -o instrumentation_data/BERT_original_combined --do-processing --title real --breakdown --throughputs

python3 proc_instru_data_bert.py \
    instrumentation_data/BERT_gen_1 \
    -o instrumentation_data/BERT_gen_1 --do-processing --title gen --throughputs

python3 proc_instru_data_dlio.py \
    instrumentation_data/BERT_benchmark_1 \
    instrumentation_data/BERT_benchmark_2 \
    bert --output instrumentation_data/BERT_benchmark_combined --title vldb --throughputs


#DLRM
python3 proc_instru_data.py \
    instrumentation_data/DLRM_original_1 \
    instrumentation_data/DLRM_original_2 \
    instrumentation_data/DLRM_original_3 \
    instrumentation_data/DLRM_original_4 \
    instrumentation_data/DLRM_original_5 \
    dlrm -o instrumentation_data/DLRM_original_combined --title real --breakdown --throughputs --latencies

python3 proc_instru_data.py \
    instrumentation_data/DLRM_gen_1 \
    instrumentation_data/DLRM_gen_2 \
    instrumentation_data/DLRM_gen_3 \
    instrumentation_data/DLRM_gen_4 \
    instrumentation_data/DLRM_gen_5 \
    dlrm -o instrumentation_data/DLRM_gen_combined --title gen --throughputs --latencies

python3 proc_instru_data.py \
    instrumentation_data/DLRM_sleep_1 \
    instrumentation_data/DLRM_sleep_2 \
    instrumentation_data/DLRM_sleep_3 \
    dlrm -o instrumentation_data/DLRM_sleep_combined --title sleep --throughputs --latencies

python3 proc_instru_data_dlio.py \
    instrumentation_data/DLRM_benchmark_1 \
    instrumentation_data/DLRM_benchmark_2 \
    instrumentation_data/DLRM_benchmark_3 \
    dlrm -o instrumentation_data/DLRM_benchmark_combined --title vldb --throughputs --latencies


