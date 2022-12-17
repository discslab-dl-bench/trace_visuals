#!/bin/bash

FILE=$1

awk '{ written[$2" "$6]+=$5; } END { for (i in written) { print i " " written[i]; }}' $FILE
