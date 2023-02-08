import os
import re
import numpy as np
from statistics import mean

from .utilities import sliding_window, get_fields, get_gpu_trace
from .mllog import get_init_start_time


def process_gpu_trace(raw_traces_dir, preproc_traces_dir, ignore_pids, UTC_TIME_DELTA):
    """
    Filters header lines, PIDs we want to ignore and events before workload initialization from gpu trace.
    Computes the average GPU utilization, writing a CSV with the values.
    """
    filter_gpu_trace(raw_traces_dir, preproc_traces_dir, ignore_pids, UTC_TIME_DELTA)
    calc_avg_gpu_usage(preproc_traces_dir)

    
def get_local_date(trace_dir) -> str:
    """
    Return the local date from the preprocessed GPU trace.
    """
    gpu_trace = get_gpu_trace(trace_dir)
    pat = re.compile(r'(^\d{4}\-\d{2}\-\d{2})T\d{2}:\d{2}:\d{2}\s+(\d+)')
    with open(gpu_trace, 'r') as gpu_trace:
        date = None
        for line in gpu_trace:
            if match := pat.match(line):
                date = match.group(1)
                print(f"Found the local date in gpu trace: {date}\n")
                break
    return np.datetime64(date)

def get_local_date_from_raw(raw_traces_dir) -> np.datetime64:
    """
    Return the local date from the nvidia-smi pmon trace in YYYY-MM-DD format.
    """
    gpu_trace = get_gpu_trace(raw_traces_dir)
    pat = re.compile(r'^\s+(\d{8})\s+(\d{2}:\d{2}:\d{2}).*')
    with open(gpu_trace, 'r') as gpu_trace:
        date = None
        for line in gpu_trace:
            if match := pat.match(line):
                date = match.group(1)
                # Convert from YYYYMMDD to YYYY-MM-DD
                date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"
                print(f"Found the local date in gpu trace: {date}\n")
                break
    return np.datetime64(date)


def get_num_gpus_from_preproc_trace(gpu_trace) -> int:
    """
    Return the number of GPUs used in the experiment.
    We assume GPUs are handed out starting from idx 0, which is the case for our workloads.
    """
    # Line format is 
    # 
    # 20230118   09:47:05      4    2647252     C     0     0     -     -   308   python 
    # or  
    # 20230118   09:46:56      5          -     -     -     -     -     -     -   -     
    #
    #  This will match only those lines with a process on GPU and have the GPU index in group 1
    pat = re.compile(r'^\d{4}\-\d{2}\-\d{2}T\d{2}:\d{2}:\d{2}\s+(\d+)')

    highest_used_gpu_idx = 0

    with open(gpu_trace, 'r') as gpu_trace:
        for line in gpu_trace:
            if match := pat.match(line):
                highest_used_gpu_idx = max(int(match.group(1)), highest_used_gpu_idx)

    return highest_used_gpu_idx + 1



def filter_gpu_trace(raw_traces_dir, preproc_traces_dir, ignore_pids, UTC_TIME_DELTA):
    """
    Filters header lines, PIDs we want to ignore and events before workload initialization from gpu trace.
    """

    init_ts = get_init_start_time(preproc_traces_dir)

    gpu_trace = os.path.join(raw_traces_dir, 'gpu.out')
    outfile = os.path.join(preproc_traces_dir, 'gpu.out')

    # Regex pattern for a line with data (vs header)
    p_dataline = re.compile(r'^\s+\d{8}\s+\d{2}:\d{2}:\d{2}\s+(\d+)\s+\d+')

    ignore = False
    if len(ignore_pids) > 0:
        ignore = True
        # Create a regex pattern from the ignore pid list
        # of the form 'pid1|pid2|pid3|...'
        p_ignore_pids = "|".join([str(pid) for pid in ignore_pids])
        p_ignore_pids = re.compile(rf'.*{p_ignore_pids}.*')

    with open(gpu_trace, 'r') as gpu_trace, open(outfile, 'w') as outfile:

        for line in gpu_trace:
            if not re.match(p_dataline, line):
                continue
            if ignore and re.match(p_ignore_pids, line):
                print(f'Ignore line: {line}')
                continue
            
            # Convert the timestamp to UTC and write out
            cols = " ".join(line.split()).replace("-", "0").split(" ")
            date = cols[0]
            ts = f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{cols[1]}"
            ts = np.datetime64(ts) + np.timedelta64(UTC_TIME_DELTA, "h")

            # In general, the GPU activity starts after initialization
            # but we'll filter out here again just in case.
            if ts >= init_ts:
                # formatted_line = f'{ts}{cols[2]:>4}{cols[3]:>12}{cols[4]:>4}{cols[5]:>5}{cols[6]:>5}{cols[7]:>5}{cols[8]:>5}{cols[9]:>8}{cols[10]:>10}\n'
                outfile.write(f'{ts} {" ".join(cols[2:])}\n')


def calc_avg_gpu_usage(preproc_traces_dir):
    """
    Computes the average GPU utilization, writing a CSV with the values.
    """
    gpu_trace = os.path.join(preproc_traces_dir, 'gpu.out')
    outcsv = os.path.join(preproc_traces_dir, "timeline", "gpu_avg.csv")

    with open(gpu_trace, 'r') as infile, open(outcsv, 'w') as outcsv:
        # Print headers
        headers = ["timestamp", "sm", "mem", "fb"]
        outcsv.write(",".join(headers) + "\n")

        line_no = 0
        num_gpus = get_num_gpus_from_preproc_trace(gpu_trace)

        print(f'Detected workload was run with {num_gpus} GPUs')

        # Slide a num_gpu wide window over the log
        # Compute the average for all columns and write out
        for line_batch in sliding_window(infile, num_gpus):

            line_no += num_gpus

            # Hold column values we care about
            wcols = []
            sm = []
            mem = []
            fb = []

            found_rank0 = False
            for i, line in enumerate(line_batch):
                try:
                    cols = get_fields(line)
                    # Keep the timestamp of the line for rank 0
                    if cols[1] == "0" and not found_rank0:
                        found_rank0 = True
                        wcols.append(cols[0])

                    # Extract values for all lines, to be averaged
                    sm.append(int(cols[4]))
                    mem.append(int(cols[5]))
                    fb.append(int(cols[8]))
                except Exception as e:
                    print(f"Exception processing GPU trace (around line {line_no + i}):\nLine: '{line}'")
                    print(e)

            # Calculate means and append to wcols
            wcols.append(str(mean(sm)))
            wcols.append(str(mean(mem)))
            wcols.append(str(mean(fb)))

            # only write to csv if saw rank 0 at least once and only once
            # there may be a sliding window frame without a rank 0 at the end
            if found_rank0:
                outcsv.write(",".join(wcols) + "\n")

