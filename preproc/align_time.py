import os
import re
import pathlib
import numpy as np
import argparse

from .mllog import get_init_start_time
from .utilities import get_fields, get_gpu_trace, get_time_align_trace
from .gpu import get_local_date_from_raw


MAX_ERR_COUNT = 5000

# We want to estimate a nsecs since boot timestamp for a specific second in local time s.t. 15:27:39.000000000. 
# This will allow us to align with some precision the timestamps with the millisecond precision UNIX timestamps 
# we get from the mllog, or other logging libraries.

# To do this, take the `trace_time_align.out` trace and look at some 1 sec transitions. Note the timestamps on 
# either side of the transition and find the smallest difference we can between timestamps. 

# Example:

# 1380327127403885  15:27:39 // dif = 368165 ns
# 1380327127772050  15:27:40 

# 1380360127432329  15:28:12 // dif = 52711 ns   -- smallest dif
# 1380360127485040  15:28:13 

# 1380328127311181  15:27:40 // dif = 254060 ns   
# 1380328127565241  15:27:41 

# 1380329127177703  15:27:41 // dif = 376942 ns
# 1380329127554645  15:27:42 

# 1380346131360213  15:27:58 // dif = 230119 ns  
# 1380346131590332  15:27:59

# The smallest is 52711 nsecs btw the timestamps for 15:28:12 and 15:28:13. 
# We can interpolate that the second change happens at the middle timestamp
# 1380360127485040 + 1380360127432329/ 2 =~ 1380360127458684

# Thus, we will estimate that 1380360127458684 ~= 15:28:13.000000000.

# Note: We see there is some drift happening since not all second changes seem to occur at the ~127msec mark in the timestamps


def _get_ref_ts(traces_dir):
    """
    This function reads the time_align trace and looks at every seconds transition
    that was captured. It looks at the difference between the nsecs timestamp before and after
    these transition and finds the minimum one.
    Using the nsecs timestamps with minimal difference, it takes their midpoint and aligns it
    with the given second in localtime. 
    """

    timealign_trace = get_time_align_trace(traces_dir)
    gpu_trace = get_gpu_trace(traces_dir)
    
    with open(timealign_trace, 'r') as timealign_trace:
        # Expected trace line format: a nsecs since boot timestamp followed by a local time
        pat = re.compile(r'(\d+)\s+(\d{2}:\d{2}:\d{2})\n')

        min_ts_diff = float('inf')
        min_lt_0 = None
        min_lt_1 = None
        
        # Discard the first lines of the trace file until we get content
        line = timealign_trace.readline()
        while not pat.match(line):
            line = timealign_trace.readline()

        # Get the initial local time
        prev_lt = pat.match(line).group(2) # local time
        # Keep iterating on the file until we find the next seconds transition
        prev_ts = pat.match(line).group(1)

        for line in timealign_trace:
            if match := pat.match(line):
                ts = match.group(1) # time stamp (nsecs since boot)
                lt = match.group(2) # local time

                # If the current line's local time is not equal to the saved one
                # then we have a seconds transition. Subtract the current timestamp
                # from the previous one and save the smallest we find.
                if lt != prev_lt:
                    diff = int(ts) - int(prev_ts)

                    if diff < min_ts_diff:
                        min_ts_diff = diff

                        min_ts_1 = ts
                        min_lt_0 = prev_lt
                        min_lt_1 = lt
                        print(f"New min timestamp diff found btw {prev_lt} and {lt}: {min_ts_diff}")

                    prev_lt = lt

                prev_ts = ts

    print(f"\nMin timestamp diff is {min_ts_diff} ns between {min_lt_0} and {min_lt_1}")

    ref_ts = int(min_ts_1) - (min_ts_diff // 2)
    ref_lt = min_lt_1

    localdate = get_local_date_from_raw(traces_dir)
    local_time_str = f"{localdate}T{ref_lt}.000000000"
    ref_local_time = np.datetime64(local_time_str)
    print(f"Alignment DONE: {ref_ts} corresponds to {ref_local_time}\n")

    return ref_ts, ref_local_time


def convert_traces_timestamp_to_UTC(traces_dir, output_dir, traces_to_align, traces_expected_cols_map, utc_timedelta):
    """
    First, interpolate an alignment between the bpftrace 'nsecs since boot' 
    timestamp and local time, then convert the timestamps to UTC.
    """

    ref_ts, ref_t = _get_ref_ts(traces_dir)

    # Gets the UTC timestamp of the INIT event in mllog
    # We want to filter out every thing before this event.
    init_ts = get_init_start_time(output_dir)

    ref_t = ref_t + np.timedelta64(utc_timedelta, "h")

    # The traces have some lines at the start where we print out columns or other info
    # We want to skip those as they don't contain useful information
    regex_start_w_number = re.compile(r'^[0-9].*')

    for trace in traces_to_align:

        print(f"Converting timestamps to UTC: {trace}")
        error_count = 0
        expected_num_cols = traces_expected_cols_map[trace]

        filename = f"trace_{trace}.out"
        tracefile = open(os.path.join(traces_dir, filename), "r")
        outfile = open(os.path.join(output_dir, f"{trace}.out"), "w")
        
        for i, line in enumerate(tracefile):
            try:
                cols = get_fields(line)
                # Handle empty lines
                if cols[0] == "":
                    # print(f"\t\t{filename} line {i} is empty. Continuing.")
                    continue

                # Handle lines that don't have the expected number of columns
                # This can occur for various reasons, often interleaving lines
                # We can have multiple acceptable number of columns as well
                if type(expected_num_cols) is list:
                    got_expected = False
                    for num_cols in expected_num_cols:
                        if len(cols) == num_cols:
                            got_expected = True
                            break

                    if not got_expected:
                        error_count += 1
                        print(f"\t\t{filename} line {i} does not have the expected number of columns. Wanted {expected_num_cols}, got {len(cols)}. Continuing.")
                        if error_count > MAX_ERR_COUNT:
                            print(f"\nERROR: More than {MAX_ERR_COUNT} errors during processing of {filename}. Aborting.")
                            print(f"You might be processing an older trace. Change the expected number of columns to match.\n")
                            exit(1)
                        continue
                else:
                    if len(cols) != expected_num_cols:
                        error_count += 1
                        print(f"\t\t{filename} line {i} does not have the expected number of columns. Wanted {expected_num_cols}, got {len(cols)}. Continuing.")
                        if error_count > MAX_ERR_COUNT:
                            print(f"\nERROR: nMore than {MAX_ERR_COUNT} errors during processing of {filename}. Aborting.")
                            print(f"You might be processing an older trace. Change the expected number of columns to match.\n")
                            exit(1)
                        continue 

                # Handle lines that don't start with a number - they should be catched by the expected
                # Number of columns check most of the time but one could slip by.
                if re.match(regex_start_w_number, line) is None:
                    print(f"\t\t{filename} line {i} does not start with a number. Continuing.")
                    continue

                # Get the timestamp
                ts = int(cols[0])
                # Calculate the diff
                ts_delta = ts - ref_ts
                # Get the UTC time
                utc_timestamp = ref_t + ts_delta

                # Only write the line if it occured after workload initialization
                if utc_timestamp >= init_ts:
                    outfile.write(np.datetime_as_string(utc_timestamp) + " " + " ".join(cols[1::]) + "\n")

            except Exception as e:
                print(f"\t\tError while processing trace_{trace}!")
                raise e


if __name__ == "__main__":

    print(f'WARNING: Ensure the traces to align, expected number of columns in each trace and UTC time delta are up to date.')
    # Add any new trace we want to time align here
    # We don't care about close, create_del for plotting (for now)
    TRACES = ["bio", "openat", "read", "write"]

    # Some lines can have an extra column or two, e.g. if reading from a file with a name vs anonymous
    TRACES_AND_EXPECTED_NUM_COLUMNS = {
        "bio": [8, 9],      # Old bio trace was 9, new one is 8
        "openat": [5, 6],   # Both 5 and 6 are valid
        "read": [5, 6],
        "write": [5, 6]     # Old trace was [8, 9]    
    }
    
    UTC_TIME_DELTA = 5

    p = argparse.ArgumentParser(description="Convert bpftrace's 'nsecs since boot' timestamps to a UTC")
    p.add_argument("traces_dir", help="Directory where raw traces are")
    p.add_argument("output_dir", help="Directory where to write the time aligned traces")
    args = p.parse_args()

    if not os.path.isdir(args.traces_dir):
        print(f"ERROR: Invalid trace directory {args.traces_dir}")
        exit(-1) 

    if not os.path.isdir(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    time_align_trace = os.path.join(args.traces_dir, "trace_time_align.out")
    if not os.path.isfile(time_align_trace):
        print(f"ERROR: Could not find trace_time_align.out in {args.traces_dir}.")
        exit(-1) 

    gpu_trace = os.path.join(args.traces_dir, "gpu.out")
    if not os.path.isfile(gpu_trace):
        print(f"ERROR: Could not find gpu.out in {args.traces_dir}")
        exit(-1) 

    convert_traces_timestamp_to_UTC(args.traces_dir, args.output_dir, TRACES, TRACES_AND_EXPECTED_NUM_COLUMNS, UTC_TIME_DELTA)

    print("All done\n")