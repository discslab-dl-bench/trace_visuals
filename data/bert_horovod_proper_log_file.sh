#!/bin/bash

# Horovod prints out some lines for each worker
# Extract those from the first rank (0) into bert.log for further processing

grep -E '\[0\]\s:::MLLOG' $1/app.log > $1/bert.log

# Remove the [0] at the start of every line that horovod adds
sed -i 's/^\[[0]\]\s//g' $1/bert.log
