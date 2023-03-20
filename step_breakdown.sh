
mkdir -p data_step_breakdown/DLRM_1gpu/raw_data

for d in $(ls $1 | grep DLRM_ )
do
    echo $d
    cp $1/${d}/dlrm.log data_step_breakdown/DLRM_1gpu/raw_data/${d}.json
done
