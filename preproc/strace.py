import os
import json
import argparse
import pathlib
import numpy as np


def convert_strace_timestamps_to_utc(strace, outdir):
    """
    Convert the UNIX timestamps of the strace log. 
    This function assumes strace was ran like `strace -ttt ...`.
    """
    infile = open(strace, "r")
    outfile = open(f"{outdir}/strace_utc.out", "w")

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
    p.add_argument("strace", help="strace output log")
    p.add_argument("outdir", help="Output directory")
    args = p.parse_args()

    if not os.path.isfile(args.strace):
        print(f"Invalid strace given")
        exit(-1) 

    if not os.path.isdir(args.outdir):
        print(f"Output dir does not exist. Creating.")
        pathlib.Path(args.data_dir).mkdir(exist_ok=True, parents=True)

    convert_strace_timestamps_to_utc(args.strace, args.outdir)
