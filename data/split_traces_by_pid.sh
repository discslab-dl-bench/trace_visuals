#!/bin/bash

if [[ $# -lt 1 || ! -d $1 ]]
then
    echo "Usage: $0 time_aligned_traces_dir"
    exit -1
fi

datadir=$1

# Create folders to house files
if [[ ! -d $datadir/bio_data ]]
then
    echo "Creating output directory $datadir/bio_data"
    mkdir -p $datadir/bio_data
fi

if [[ ! -d $datadir/open_data ]]
then
    echo "Creating output directory $datadir/open_data"
    mkdir -p $datadir/open_data
fi

if [[ ! -d ./read_data ]]
then
    echo "Creating output directory $datadir/read_data"
    mkdir -p $datadir/read_data
fi

if [[ ! -d $datadir/write_data ]]
then
    echo "Creating output directory $datadir/write_data"
    mkdir -p $datadir/write_data
fi

# Remove empty lines from the pids
sed -i '/^[[:space:]]*$/d' $datadir/pids

echo "PIDs to be extracted:"
cat $datadir/pids

# Use the PIDs to extract their lines from each trace
while read pid; do
    # Remove whitespaces and newlines from pid
    pid=${pid//[$'\t\r\n']}
    echo "Processing $pid"

    # Block IO trace
    # Extract the pid's lines from the trace, remove less relevant columns
    # We're keeping only timestamp, pid, r/w, size, latency.
    grep -a " $pid " $1/bio_time_aligned.out | awk -F '\\s+' '{print $1,$2,$5,$6,$8}' > "$datadir/bio_data/bio_$pid"
    # grep -a " $pid " $1/bio_time_aligned.out | awk -F '\\s+' '{print $1,$2,$7,$8,$9}' > "$datadir/bio_data/bio_$pid"    # Old trace
    # grep -a " $pid " $1/bio_time_aligned.out | awk -F '\\s+' '{print $1,$2,$7,$9,$10}' > "$datadir/bio_data/bio_$pid"    # Even Older trace

    grep -a " $pid " $1/openat_time_aligned.out | awk -F ' ' '{print $1,$2,$4,$5,$6}' > "$datadir/open_data/open_$pid"

    grep -a " $pid " $1/read_time_aligned.out | awk -F ' ' '{print $1,$2,$4,$5,$6}' > "$datadir/read_data/read_$pid"

    grep -a " $pid " $1/write_time_aligned.out | awk -F ' ' '{print $1,$2,$4,$5}' > "$datadir/write_data/write_$pid"
    # grep -a " $pid " $1/write_time_aligned.out | awk -F ' ' '{print $1,$2,$3,$8}' > "$datadir/write_data/write_$pid"        # Old write trace


done < $datadir/pids

echo -e "\nAll done\n"
