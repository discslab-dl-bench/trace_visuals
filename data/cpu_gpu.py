import os
import argparse
import numpy as np
from statistics import mean
from datetime import datetime
from itertools import zip_longest as izip_longest

# We are in eastern time, so UTC-4
UTC_TIME_DELTA = 4

def sliding_window(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def calc_avg_gpu_usage(gpu_trace, num_gpus):

    data_dir = os.path.dirname(gpu_trace)

    infile = open(gpu_trace, "r")
    outcsv = open(os.path.join(data_dir, "gpu_avg.csv"), "w")

    # Print headers
    headers = ["timestamp", "sm", "mem", "fb"]
    outcsv.write(",".join(headers) + "\n")

    line_no = 0

    # Read the file num_gpus lines at a time. 
    # Compute the average for all columns and write out
    for line_batch in sliding_window(infile, num_gpus):

        line_no += 8

        # Hold column values we care about
        wcols = []
        sm = []
        mem = []
        fb = []

        first_rank0 = False # flag sets to true if we encountered rank 0 at least once and only once
        for i, line in enumerate(line_batch):
            try:
                cols = " ".join(line.split()).replace("-", "0").split(" ")
                if cols[2] == "0" and not first_rank0:
                    # Combine cols 0 and 1 into a UTC timestamp for every $num_gpus (ex: 8, 4, ...) lines
                    date = cols[0]
                    ts = f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{cols[1]}"
                    ts = str(np.datetime64(ts) + np.timedelta64(UTC_TIME_DELTA, "h"))
                    wcols.append(ts)
                    first_rank0 = True
                # Extract values
                sm.append(int(cols[5]))
                mem.append(int(cols[6]))
                fb.append(int(cols[9]))
            except Exception as e:
                print(f"aruond line {line_no + i}")
                print(e)
                print(line)

        # Calculate means and append to wcols
        wcols.append(str(mean(sm)))
        wcols.append(str(mean(mem)))
        wcols.append(str(mean(fb)))

        # only write to csv if saw rank 0 at least once and only once
        if first_rank0:
            outcsv.write(",".join(wcols) + "\n")

    infile.close()
    outcsv.close()


def get_date(gpu_trace):
    """
    Returns the date of the traces in YYYY-MM-DD format 
    """
    infile = open(gpu_trace, "r")
    for line in infile:
        cols = " ".join(line.split()).replace("-", "0").split(" ")
        if cols[2] == "0":
            # Combine cols 0 and 1 into a UTC timestamp
            date = cols[0]
            date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
            return np.datetime64(date)


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


def process_cpu_data(cpu_trace, current_date):
    """
    Convert raw cpu usage data to a csv file
    """

    data_dir = os.path.dirname(cpu_trace)

    infile = open(cpu_trace, "r")
    outcsv = open(os.path.join(data_dir, "cpu_all.csv"), "w")

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

    for line in infile:
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

        # Make UTC timestamp from time and current date
        cols[0] = str(np.datetime64(str(current_date) + "T" + cols[0]) + np.timedelta64(UTC_TIME_DELTA, "h"))

        # Join the columns with commas and write to the CSV file
        outcsv.write(",".join(cols) + "\n")

    infile.close()
    outcsv.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Outputs CSV files of average GPU and CPU use.")
    p.add_argument("gpu_trace", help="The nvidia-smi GPU trace")
    p.add_argument("cpu_trace", help="The mpstat CPU trace, containing only the lines for 'all'")
    p.add_argument("num_gpus", type=int, help="The number of GPUs used for training")
    args = p.parse_args()

    if not os.path.isfile(args.gpu_trace):
        print(f"Invalid GPU trace file given")
        exit(-1) 

    if not os.path.isfile(args.cpu_trace):
        print(f"Invalid CPU trace file given")
        exit(-1) 

    print("Calculating average GPU usage")
    calc_avg_gpu_usage(args.gpu_trace, args.num_gpus)

    print("Processing CPU data")
    current_date = get_date(args.gpu_trace)
    process_cpu_data(args.cpu_trace, current_date)

    print("All done\n")



