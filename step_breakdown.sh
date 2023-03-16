

mkdir -p data_step_breakdown/UNET_2/raw_data

for d in $(ls $1)
do
    echo $d
    cp $1/${d}/unet3d.log data_step_breakdown/UNET_2/raw_data/${d}.json
done