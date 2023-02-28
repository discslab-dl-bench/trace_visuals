import sys
import json
import statistics

from preproc.utilities import get_fields

def max_read_size(file, skip_epoch_1=False):

    max_read = 0
    max_line = ""

    with open(file, 'r') as file:

        for line in file:

            data = get_fields(line)

            read_size = int(data[3])

            if read_size > max_read:
                max_read = read_size
                max_line = line

    print(f'Max read line: {max_line}')
    
    # print(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}")
    # for key in all_times:
    #     avg = round(statistics.mean(all_times[key]), 4)
    #     median = round(statistics.median(all_times[key]), 4)
    #     std = round(statistics.stdev(all_times[key]), 4)
    #     quantiles = statistics.quantiles(all_times[key])

    #     print(f"{key:>30}:\t{avg:>15}\t{median:>15}\t{std:>15}\t{round(quantiles[0], 4):>15}\t{round(quantiles[2], 4):>15}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} trace_read.out")
        exit(1)

    max_read_size(sys.argv[1], True)