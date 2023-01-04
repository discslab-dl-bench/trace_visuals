import argparse
from pathlib import Path

from preproc.align_time import *

# We are in eastern time, so UTC-4 during Daylight Savings Time and UTC-5 otherwise
UTC_TIME_DELTA = 5

# Add any new trace we want to plot here
# We don't care about close, create_del for plotting (for now)
TRACES = ["bio", "openat", "read", "write"]

# Some lines can have an extra column or two, e.g. if reading from a file with a name vs anonymous
# Change this if processing old / newer traces with different fields.
TRACES_AND_EXPECTED_NUM_COLUMNS = {
    "bio": [8, 9],      # Old bio trace was 9, new one is 8
    "openat": [5, 6],   # Both 5 and 6 are valid
    "read": [5, 6],
    "write": [5, 6]     # Old trace was [8, 9]    
}


def preprocess_traces(args):
    align_time(args)


def validate_args(args):
    if not os.path.isdir(args.traces_dir):
        print(f"ERROR: Invalid trace directory {args.traces_dir}")
        exit(-1) 
    if not os.path.isdir(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Convert bpftrace's 'nsecs since boot' timestamps to a UTC")
    p.add_argument("traces_dir", help="Raw traces directory")
    p.add_argument("-o", "--output-dir", help="Output directory")
    args = p.parse_args()

    if not args.output_dir:
        dir = Path(args.traces_dir)
        trace_name = dir.name
        args.output_dir = dir.parent / f"ta_{dir.name}"
        print(args.output_dir)
        
    validate_args(args)

    preprocess_traces(args)