import sys
import json
import statistics

def average_times(file, skip_epoch_1=False):

    all_times = {}

    with open(file, 'r') as log:

        for line in log:
            # discard the prefix
            line = line.replace(":::MLLOG ", "")

            # load as now valid json
            line = json.loads(line)

            if line['value'] is None:
                continue
            
            if not isinstance(line['value'], dict):
                continue

            if skip_epoch_1 and 'epoch_num' in line['metadata'] and line['metadata']['epoch_num'] == 1:
                continue

            key = line['key']

            if 'duration' in line['value']:
                time = line['value']['duration'] / 1_000_000_000
            else:
                continue

            if key not in all_times:
                all_times[key] = [time]
            else:
                all_times[key].append(time)

    
    print(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}")
    for key in all_times:
        avg = round(statistics.mean(all_times[key]), 4)
        median = round(statistics.median(all_times[key]), 4)
        std = round(statistics.stdev(all_times[key]), 4)
        quantiles = statistics.quantiles(all_times[key])

        print(f"{key:>30}:\t{avg:>15}\t{median:>15}\t{std:>15}\t{round(quantiles[0], 4):>15}\t{round(quantiles[2], 4):>15}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} unet3d.log")
        exit(1)

    average_times(sys.argv[1], True)