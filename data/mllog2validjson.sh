#!/bin/bash

parent=$(dirname $1)
parent2=$(basename $parent)

logfile=$(basename $1) 

# Remove ":::MLLOG" prefix from all lines
sed 's/:::MLLOG //' ${parent}/${logfile} >  ${parent}/${logfile}_valid

# Remove empty namespace field
awk -F ', ' 'BEGIN { OFS= ", "; ORS="\n"} {$1="{"; print $0}' ${parent}/${logfile}_valid > tmp && mv tmp ${parent}/${logfile}_valid
sed -i 's/{, /{/' ${parent}/${logfile}_valid

awk 'BEGIN { print "[" } { print $0"," }' ${parent}/${logfile}_valid  > tmp && mv tmp ${parent}/${logfile}_valid
# Remove last comma, make valid JSON array
sed -i '$ s/.$/\n]/' ${parent}/${logfile}_valid
mv ${parent}/${logfile}_valid train_times/raw_data/${parent2}.json
