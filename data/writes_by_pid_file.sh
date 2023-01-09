#!/bin/bash

FILE=$1

DIRNAME=$(dirname $1)

# awk '{ amt_written[$2" "$6]+=$4; time_writing[$2" "$6]+=$5;} END { for (i in amt_written) { print i " " amt_written[i]; }}' $FILE
# awk '(NR > 2) { amt_written[$2" "$6]+=$4; time_writing[$2" "$6]+=$5;} END { print "\nAmounts written\n"; for (i in amt_written) { print i " " amt_written[i] " B"}; print "\nTime Writing\n"; for (i in time_writing) {print i " " time_writing[i] / 1000000, " ms"; }}' $FILE

echo -e "Amounts Written\n"
awk '{ amt_written[$2"\t"$6]+=$4; } END { for (i in amt_written) { print i " " amt_written[i] " B"; }}' $FILE | sort

echo -e "\nTime Writing\n"
awk '{ time_writing[$2"\t"$6]+=$5; } END { for (i in time_writing) {print i " " time_writing[i] / 1000000, "ms"; }}' $FILE | sort



