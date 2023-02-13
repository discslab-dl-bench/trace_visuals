import os
import re
import argparse
import numpy as np
from datetime import datetime

from preproc.mllog import get_init_start_time

from .utilities import get_cpu_trace
from .gpu import get_local_date_from_raw


def split_cpu_trace_line_cols(line):
    # Remove duplicate spaces, and split
    cols = " ".join(line.replace("all", "").split()).split(" ")

    # Depending on the time config of the machine, mpstat
    # may output an AM/PM timestamp or a 24h timestamp.
    # Always return 24h timestamp in col[0] and remove the AM/PM column.
    if cols[1] == "AM" or cols[1] == "PM":
        timestamp = f"{cols[0]} {cols[1]}"
        # This converts an AM/PM timestamp to 24h
        time_AM_PM = datetime.strptime(timestamp, "%I:%M:%S %p")
        time_24h = datetime.strftime(time_AM_PM, "%H:%M:%S")
        cols[0] = time_24h
        cols.remove(cols[1]) # Delete the AM/PM column

    return cols


def process_cpu_trace(raw_traces_dir, preproc_traces_dir, UTC_TIME_DELTA):
    """
    Convert raw cpu usage data to a csv file, filtering out events before initialization
    """
    print('Processing CPU trace')
    current_date = get_local_date_from_raw(raw_traces_dir)
    init_ts = get_init_start_time(preproc_traces_dir)

    cpu_trace = get_cpu_trace(raw_traces_dir)
    oututc = os.path.join(preproc_traces_dir, "cpu.out")
    outcsv = os.path.join(preproc_traces_dir, "timeline", "cpu_all.csv")

    # This trace is a bit more complicated, because mpstat has different outputs on different machines
    # E.g. sometimes it will have an AM/PM field, other times will use a 24h clock

    with open(cpu_trace, "r") as infile, open(outcsv, 'w') as outcsv, open(oututc, 'w') as oututc:
        # Print headers
        headers = [
            "timestamp",
            "%usr",
            "%nice",
            "%sys",
            "%iowait",
            "%irq",
            "%soft",
            "%steal",
            "%guest",
            "%gnice",
            "%idle",
        ]
        outcsv.write(",".join(headers) + "\n")

        date_changed = False

        for i, line in enumerate(infile):
            
            if len(line.strip()) == 0:
                continue

            # Remove duplicate spaces, and split
            cols = split_cpu_trace_line_cols(line)

            # Don't process the last line
            if cols[0] == "Average:":
                break

            # Increment the date by one if we traced past midnight
            # This check assumes we have a line for each second
            if not date_changed and cols[0] == "00:00:00":
                current_date += 1
                date_changed = True

            try:
                # Make UTC timestamp from time and current date
                ts = np.datetime64(str(current_date) + "T" + cols[0]) + np.timedelta64(UTC_TIME_DELTA, "h")
                if ts >= init_ts:
                    cols[0] = str(ts)
                    # Join the columns with commas and write to the CSV file
                    outcsv.write(",".join(cols) + "\n")
                    # Write a copy of the trace with UTC timestamp, might be useful for debugging
                    oututc.write(" ".join(cols) + '\n')
            except:
                print(f"Skipping line: {line}")
                continue


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Outputs CSV files of average GPU and CPU use.")
    p.add_argument("raw_traces_dir", help="Directory of raw trace data")
    p.add_argument("ta_traces_dir", help="Directory of time aligned traces")
    args = p.parse_args()

    if not os.path.isdir(args.raw_traces_dir):
        print(f"Invalid raw traces directory given")
        exit(-1) 

    if not os.path.isdir(args.ta_traces_dir):
        print(f"Invalid raw traces directory given")
        exit(-1) 

    current_date = get_local_date_from_raw(args.raw_traces_dir)

    cpu_trace = os.path.join(args.ta_traces_dir, "cpu_data", "cpu.all")


    print("Processing CPU data")
    process_cpu_trace(cpu_trace, current_date)

    print("All done\n")



