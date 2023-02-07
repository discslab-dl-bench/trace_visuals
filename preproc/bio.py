import os
import argparse
import statistics

from .utilities import plot_histogram, get_fields

# Column index of latency in the bio trace.
# Latest columns are
# TIMESTAMP         PID      COMM      DISK  T   BYTES    SECTOR    LAT(ns)
COLUMN_INDEX = 7

def _get_p99_latency(bio_trace, traces_dir) -> int:
    all_latencies = []
    with open(bio_trace, "r") as tracefile:
        for line in tracefile:
            cols = get_fields(line)
            latency = int(cols[COLUMN_INDEX])
            all_latencies.append(latency)

    plot_histogram(all_latencies, traces_dir, "BIO latencies", "bio_latencies_raw", nbins=500)
    pctls = statistics.quantiles(all_latencies, n=100)
    p99 = int(pctls[98])
    return p99


def process_long_bio_calls(bio_trace, traces_dir, verbose=False) -> None:
    """
    Go through the bio trace and detect/remove all bio calls above p99 latency.
    We do this for timeline plot legibility.
    The original trace is backed-up.
    """
    tmp_out = bio_trace + "_tmp"

    p99 = _get_p99_latency(bio_trace, traces_dir)

    all_latencies = []
    lines_removed = 0

    with open(bio_trace, "r") as tracefile, open(tmp_out, "w") as outfile:
        for i, line in enumerate(tracefile):
            cols = get_fields(line)
            latency = int(cols[COLUMN_INDEX])
            
            # Write back out all trace lines except those above p99 latency
            if latency > p99:
                lines_removed += 1
                if verbose:
                    print(f"{cols[0]} (line {i}): Long latency of {latency:,} ns")
            else:
                all_latencies.append(latency)
                outfile.write(line)

    plot_histogram(all_latencies, traces_dir, "BIO latencies", "bio_latencies", nbins=500)

    # Keep a backup of the original
    # Rename tmp to replace original
    os.rename(bio_trace, f"{bio_trace}.bk")
    os.rename(tmp_out, bio_trace)


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Go through the bio trace and remove all operations above p99 latency.")
    p.add_argument("traces_dir", help="Time aligned traces directory")
    p.add_argument("-v", "--verbose", action='store_true', help="Print out each detected long bio call")
    args = p.parse_args()

    if not os.path.isdir(args.traces_dir):
        print(f"ERROR: Invalid trace directory {args.traces_dir}")
        exit(-1) 

    bio_trace = os.path.join(args.traces_dir, "bio_UTC.out")
    if not os.path.isfile(bio_trace):
        print(f"ERROR: Invalid bio trace {bio_trace}")
        exit(-1) 

    process_long_bio_calls(bio_trace, args.traces_dir, verbose=args.verbose)
