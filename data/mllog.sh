#!/bin/bash

if [ $# -lt 2 ]
then
    echo "Usage: $0 unet3d.log output_dir workload"
    exit -1
fi

logfile=$1
output_dir=$2
workload=$3

if [[ ! -d $output_dir/mllog_data ]]
then
    echo "Creating output directory $output_dir/mllog_data"
    mkdir $output_dir/mllog_data
fi

output_dir=$output_dir/mllog_data

# Remove ":::MLLOG" prefix from all lines
sed 's/:::MLLOG //' $logfile > $output_dir/u.log

# Remove empty namespace field
awk -F ', ' 'BEGIN { OFS= ", "; ORS="\n"} {$1="{"; print $0}' $output_dir/u.log > tmp && mv tmp $output_dir/u.log
sed -i 's/{, /{/' $output_dir/u.log

# Extract training timeline info 
# The log will contain different events depending on the workload

if [[ $workload == "unet3d" ]]
then
    grep -Ea "init_start|init_stop|epoch_start|epoch_stop|eval_start|eval_stop|checkpoint_start|checkpoint_stop" $output_dir/u.log > $output_dir/timeline.log
elif [[ $workload == "dlrm" ]]
then
    grep -Ea "init_start|init_stop|block_start|block_stop|eval_start|eval_stop|training_start|training_stop|checkpoint_start|checkpoint_stop" $output_dir/u.log > $output_dir/timeline.log
elif [[ $workload == "bert" ]]
then
    grep -Ea "init_start|init_stop|block_start|block_stop|checkpoint_start|checkpoint_stop" $output_dir/u.log > $output_dir/timeline.log
else
    echo "Unknown workload $workload"
    exit
fi

# sed -i '$ d' $output_dir/timeline.log

awk 'BEGIN { print "[" } { print $0"," }' $output_dir/timeline.log > tmp && mv tmp $output_dir/timeline.log
# Remove last comma, make valid JSON array
sed -i '$ s/.$/\n]/' $output_dir/timeline.log

# rm $output_dir/u.log

echo -e "All done\n"