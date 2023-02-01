import os
import re
import pathlib
import numpy as np
import argparse

# We are in eastern time, so UTC-4
UTC_TIME_DELTA = 4

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

"""
This function reads the time align trace and looks at every seconds transition
that was captured. It looks at the difference between the nsecs timestamp before and after
these transition and finds the minimum one.
Using the nsecs timestamps with minimal difference, it takes their midpoint and aligns it
with the given second in localtime. 
"""
def get_ref_ts(timealign_trace, gpu_trace):

    timealign_trace = open(timealign_trace, "r")
    
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

                    min_ts_0 = prev_ts
                    min_ts_1 = ts
                    min_lt_0 = prev_lt
                    min_lt_1 = lt
                    print(f"New min timestamp diff found btw {prev_lt} and {lt}: {min_ts_diff}")

                prev_lt = lt

            prev_ts = ts

    print(f"\nMin timestamp diff is {min_ts_diff} ns between {min_lt_0} and {min_lt_1}")

    ref_ts = int(min_ts_1) - (min_ts_diff // 2)
    ref_lt = min_lt_1

    # Try to get the date from the GPU trace
    gpu_trace = open(gpu_trace, "r")

    pat = re.compile(r'^\s+(\d{8})\s+(\d{2}:\d{2}:\d{2}).*')

    localdate = None
    for line in gpu_trace:
        if match := pat.match(line):
            localdate = match.group(1)
            print(f"Found the local date in gpu trace: {localdate}\n")
            break

    # Gpu trace localdate is in YYYYMMDD, break it into YYYY-MM-DD
    local_time_str = f"{localdate[0:4]}-{localdate[4:6]}-{localdate[6:]}T{ref_lt}.000000000"
    ref_local_time = np.datetime64(local_time_str)
    print(f"Alignment DONE: {ref_ts} corresponds to {ref_local_time}\n")

    return ref_ts, ref_local_time

# Read a line and revert the file pointer 
def peek_line(f):
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    return line

"""
Once we have estimated the matching between nsecs timestamp and local time, we can  
convert the timestamps to UTC to be able to align them with data from other sources.
"""
def align_all_traces(traces_dir, output_dir, ref_ts, ref_t):

    print("Aligning all traces:")

    ref_t = ref_t + np.timedelta64(UTC_TIME_DELTA, "h")

    # The traces have some lines at the start where we print out columns or other info
    # We want to skip those as they don't contain useful information
    regex_start_w_number = re.compile(r'^[0-9].*')

    for trace in TRACES:

        print(f"\tProcessing {trace}")
        error_count = 0
        expected_num_cols = TRACES_AND_EXPECTED_NUM_COLUMNS[trace]

        filename = "trace_" + trace + ".out"
        tracefile = open(os.path.join(traces_dir, filename), "r")
        outfile = open(os.path.join(output_dir, trace + "_time_aligned.out"), "w")
        
        for i, line in enumerate(tracefile):
            try:
                cols = " ".join(line.split()).split(" ")
                # Handle empty lines
                if cols[0] == "":
                    print(f"\t\t{filename} line {i} is empty. Continuing.")
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
                        if error_count > 10000:
                            print(f"\nERROR: More than 100 errors during processing of {filename}. Aborting.")
                            print(f"Most likely you're processing an older trace. Change the expected number of columns to match.\n")
                            exit(1)
                        continue
                else:
                    if len(cols) != expected_num_cols:
                        error_count += 1
                        print(f"\t\t{filename} line {i} does not have the expected number of columns. Wanted {expected_num_cols}, got {len(cols)}. Continuing.")
                        if error_count > 10000:
                            print(f"\nERROR: nMore than 100 errors during processing of {filename}. Aborting.")
                            print(f"Most likely you're processing an older trace. Change the expected number of columns to match.\n")
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
                t = ref_t + ts_delta
                outfile.write(np.datetime_as_string(t) + " " + " ".join(cols[1::]) + "\n")

            except Exception as e:
                print(f"\t\tError while processing trace_{trace}! Continuing.")
                print(e)
                continue


if __name__ == "__main__":

    print('#########################################################################')
    print("align_time.py: Converting bpftrace's 'nsecs since boot' timestamps to UTC")
    print('#########################################################################\n')

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

    ref_ts, ref_local_time = get_ref_ts(time_align_trace, gpu_trace)
    align_all_traces(args.traces_dir, args.output_dir, ref_ts, ref_local_time)

    print("All done\n")