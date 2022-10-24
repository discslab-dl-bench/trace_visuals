#!/bin/bash

# This script will preprocess and plot raw traces data

# Determine which python command to use depending on platform 
unameOut="$(uname -s)"
case "${unameOut}" in
    MINGW*)     py=py;;
    *)          py=python3;;
esac

if [ $# -ne 2 ]
then
    echo "Usage: $0 <raw data dir> <num gpus used>" 
    exit -1
fi

datadir=$1
num_gpus=$2
expname=$(basename $datadir)
datadirname=$(dirname $datadir)
# extrace 'data/' since we'll be moving into it
datadir_relative=${datadir#data/}
echo "Datadir relative $datadir_relative"

# ta_dir=$datadir/raid/data/dlrm/dlrm_tracing/terabyte_tracing_20221001223330ame/ta_$expname
ta_dir=$datadirname/ta_$expname

pushd data
./preprocess_traces.sh $datadir_relative $num_gpus
popd
${py} timeline.py $ta_dir $expname