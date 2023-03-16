#!/bin/bash

python3 step_breakdown_dlio.py dlio_unet_instru unet3d
python3 step_breakdown_dlio.py dlio_bert_instru bert
python3 step_breakdown_dlio.py dlio_dlrm_instru dlrm
