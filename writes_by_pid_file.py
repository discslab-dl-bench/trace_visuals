import os
import argparse
import pathlib


"""
As of time of writing, the format for both the read and write traces were:
TIMESTAMP   PID     OFFSET      SIZE (B)    LAT (ns)    FILENAME
"""
def analyze_trace(tracefile):

    with open(tracefile, mode='r') as trace:

        for line in trace:
            cols = " ".join(line.split()).split(" ")

            

    pass



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyzes the reads/writes contained in tracefile")
    p.add_argument("tracefile", help="read or write trace")
    args = p.parse_args()

    if not os.path.isfile(args.tracefile):
        print(f"Invalid tracefile given")
        exit(-1)

    analyze_trace(args.tracefile)
