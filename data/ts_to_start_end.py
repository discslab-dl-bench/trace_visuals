import os
import sys
import numpy as np
from pathlib import Path


def main(filename, outdir):

    pid = os.path.basename(filename).split("_")[1]

    infile = open(filename, "r")
    outfile = open(f"{outdir}/st_end_data_{pid}", "w")

    for i, line in enumerate(infile):
        try:
            cols = line.split(",")
            ts_end = np.datetime64(cols[0])
            lat = int(cols[2])
            ts_start = ts_end - lat
            outfile.write(f"{ts_start},{ts_end},{cols[1]}\n")
        except Exception as e:
            print(f"\t{filename} line {i}: {e}. Continuing.")
            continue


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} filename")
        exit(1)

    filename = sys.argv[1]
    # Get parent directory
    outdir = os.path.dirname(os.path.dirname(filename))
    outdir = os.path.join(outdir, "st_end_data")

    print(outdir)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    main(filename, outdir)
