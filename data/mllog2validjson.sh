#!/bin/bash

if [ $# != 2 ]
then
    echo "Usage: $0 unet3d.log experiment_name"
    exit 1
fi

logfile=$(basename $1)
parent=$(dirname $1)
exp_name=$(basename $parent)

outdir=train_times/${2}/raw_data
mkdir -p $outdir

# Remove ":::MLLOG" prefix from all lines
sed 's/:::MLLOG //' ${parent}/${logfile} >  ${parent}/${logfile}_valid.json

# Remove empty namespace field
awk -F ', ' 'BEGIN { OFS= ", "; ORS="\n"} {$1="{"; print $0}' ${parent}/${logfile}_valid.json > tmp && mv tmp ${parent}/${logfile}_valid.json
sed -i 's/{, /{/' ${parent}/${logfile}_valid.json

awk 'BEGIN { print "[" } { print $0"," }' ${parent}/${logfile}_valid.json  > tmp && mv tmp ${parent}/${logfile}_valid.json
# Remove last comma, make valid JSON array
sed -i '$ s/.$/\n]/' ${parent}/${logfile}_valid.json

cp ${parent}/${logfile}_valid.json ${outdir}/${exp_name}.json
