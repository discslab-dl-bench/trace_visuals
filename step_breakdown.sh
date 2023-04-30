# Little script to move the processed app logs to a folder for step breakdwon processing
mkdir -p data_step_breakdown/DLRM_instru_4/raw_data

for d in $( ls data/mar16/DLRM_instru_4 )
do
    echo $d
    cp data/mar16/DLRM_instru_4/${d}/processed/dlrm.log data_step_breakdown/DLRM_instru_4/raw_data/${d}.json
done
