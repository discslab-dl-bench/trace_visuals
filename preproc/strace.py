import os
import json
import argparse
import pathlib
import numpy as np
from preproc.gpu import get_local_date
from preproc.utilities import get_strace_trace



def convert_strace_tt_timestamps_to_utc(traces_dir, outdir, UTC_TIME_DELTA):
    """
    Convert the UNIX timestamps of the strace log. 
    This function assumes strace was ran like `strace -tt ...`.
    """
    strace_trace = get_strace_trace(traces_dir)

    local_date = get_local_date(outdir)

    with open(strace_trace, "r") as infile, open(f"{outdir}/strace.out", "w") as outfile:
        for i, line in enumerate(infile):
            
            cols = " ".join(line.split()).split(" ")
            # Handle empty lines
            if cols[0] == "":
                print(f"\t line {i} is empty. Continuing.")
                continue

            ts = np.datetime64(f'{local_date}T{cols[1]}') + np.timedelta64(UTC_TIME_DELTA, "h")
            cols[1] = np.datetime_as_string(ts)
            outfile.write(" ".join(cols) + "\n")


def convert_strace_ttt_timestamps_to_utc(traces_dir, outdir):
    """
    Convert the UNIX timestamps of the strace log. 
    This function assumes strace was ran like `strace -ttt ...`.
    """
    strace_trace = get_strace_trace(traces_dir)

    with open(strace_trace, "r") as infile, open(f"{outdir}/strace.out", "w") as outfile:
        for i, line in enumerate(infile):
            
            cols = " ".join(line.split()).split(" ")
            # Handle empty lines
            if cols[0] == "":
                print(f"\t line {i} is empty. Continuing.")
                continue

            # The strace UNIX timestamps are in microseconds and 
            # have a decimal that must be removed
            ts = np.datetime64(int(cols[1].replace(".", "")), "us")
            cols[1] = np.datetime_as_string(ts)

            outfile.write(" ".join(cols) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Changes the UNIX timestamp in an strace log to a UTC timestamp")
    p.add_argument("trace_dir", help="strace output log")
    p.add_argument("outdir", help="Output directory")
    args = p.parse_args()

    if not os.path.isdir(args.trace_dir):
        print(f"Invalid trace_dir given")
        exit(-1) 

    if not os.path.isdir(args.outdir):
        print(f"Output dir does not exist. Creating.")
        pathlib.Path(args.data_dir).mkdir(exist_ok=True, parents=True)

    convert_strace_tt_timestamps_to_utc(args.strace, args.outdir)
