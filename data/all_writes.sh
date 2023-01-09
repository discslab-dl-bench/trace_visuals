declare -a num_gpus=(2 4 6 8)

for num_gpu in "${num_gpus[@]}";
do  
    for batch_size in $(seq 1 5)
    do
        dirname="ta_UNET_${num_gpu}GPU_batch${batch_size}_instrumented"

        ./writes_by_pid_file.sh ${dirname}/write_time_aligned.out > ${dirname}/writes_analysis.txt
        ./writes_by_pid_file.sh ${dirname}/read_time_aligned.out > ${dirname}/reads_analysis.txt
    done
done
