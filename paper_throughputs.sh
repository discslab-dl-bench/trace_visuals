#!/bin/bash

# Generate the step breakdown and throughput plots

# UNET3D
python3 step_breakdown.py data_step_breakdown/UNET_2 data_step_breakdown/UNET_bugfix data_step_breakdown/UNET_instru_mar20 unet3d -o data_step_breakdown/UNET_combined_3 --breakdown --throughputs --latencies
python3 step_breakdown.py data_step_breakdown/UNET3D_gen data_step_breakdown/UNET3D_gen_2 unet3d --title gen -o data_step_breakdown/UNET3D_gen_combined --throughputs --latencies
python3 step_breakdown.py data_step_breakdown/UNET3D_sleep_3 data_step_breakdown/UNET3D_sleep_4 unet3d --title sleep -o data_step_breakdown/UNET3D_sleep_combined --throughputs --latencies
python3 step_breakdown_dlio.py data/mar16/DLIO_UNET_formula data/mar16/DLIO_UNET_instru_3 data/mar16/DLIO_UNET_instru_4 data/mar16/DLIO_UNET_preproc unet3d -o data/mar16/DLIO_UNET_combined --title vldb --throughputs --latencies

#BERT
python3 step_breakdown_bert.py BERT_profiler_2 BERT_profiler --do-processing --title real --breakdown --throughputs
python3 step_breakdown_bert.py data/mar16/BERT_gen -o data/mar16/BERT_gen --do-processing --title gen --throughputs
python3 step_breakdown_dlio.py data/mar15/DLIO_BERT_instru_2 data/mar16/DLIO_BERT_formula bert --output data/mar15/DLIO_BERT_combined_2 --title vldb --throughputs

#DLRM
python3 step_breakdown.py data_step_breakdown/DLRM_instru_mar20 data_step_breakdown/DLRM_instru_4 data_step_breakdown/DLRM_instru_5 data_step_breakdown/DLRM_instru_6 data_step_breakdown/DLRM_instru_7 dlrm -o data_step_breakdown/DLRM_combined_3 --title real --breakdown --throughputs --latencies
python3 step_breakdown.py data_step_breakdown/DLRM_gen_1 data_step_breakdown/DLRM_gen_2 data_step_breakdown/DLRM_gen_3 data_step_breakdown/DLRM_gen_4 data_step_breakdown/DLRM_gen_5 dlrm -o data_step_breakdown/DLRM_gen_combined_2 --title gen --throughputs --latencies
python3 step_breakdown.py data_step_breakdown/DLRM_sleep_3 data_step_breakdown/DLRM_sleep_4 dlrm -o data_step_breakdown/DLRM_sleep_combined_2 --title sleep--throughputs --latencies
python3 step_breakdown_dlio.py data/mar16/DLIO_DLRM_formula data/mar16/DLIO_DLRM_formula_2 data/mar16/DLIO_DLRM_formula_3 dlrm -o data/mar16/DLIO_DLRM_combined_formula_3 --title vldb --throughputs --latencies


