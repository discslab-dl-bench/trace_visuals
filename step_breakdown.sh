

mkdir -p data_step_breakdown/DLRM_LARGE_2/raw_data

for d in $(ls $1)
do
    echo $d
    cp $1/${d}/dlrm.log data_step_breakdown/DLRM_LARGE_2/raw_data/${d}.json
done