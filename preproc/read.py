import os
import argparse
import statistics

from .utilities import get_read_trace, plot_histogram, get_fields

# Column index of latency in the bio trace.
# Latest columns are
# TIMESTAMP         PID    OFFSET   RET (B)      LAT(ns)      FILENAME
READ_SZ_IDX = 3
FILENAME_IDX = 5

def process_absurdly_large_read_sizes(traces_dir, verbose=False) -> None:
    """
    We sometimes get some absurdly long returned read sizes in the read trace
    when reading /sys/class/net/eth0/speed. Remove them.
    TIMESTAMP         PID      OFFSET   RET (B)               LAT(ns)      FILENAME
    134078025034623   1797499  0x0      18446744073709551103  14865        speed
    134078025170518   1797505  0x0      18446744073709551103  14637        speed
    """
    print('Removing absurdly large reads ( > 10^18 B) from read trace.')
    read_trace = get_read_trace(traces_dir)
    tmp_out = read_trace + "_tmp"

    all_read_sizes = []
    lines_removed = 0

    with open(read_trace, "r") as tracefile, open(tmp_out, "w") as outfile:
        for i, line in enumerate(tracefile):
            cols = get_fields(line)
            read_size = int(cols[READ_SZ_IDX])
            
            # 10^15 
            if read_size > 1_000_000_000_000_000:
                lines_removed += 1
                if verbose:
                    print(f"{cols[0]} (line {i}): Absurd read size of of {read_size:,} B")
            else:
                all_read_sizes.append(read_size)
                outfile.write(line)

    plot_histogram(all_read_sizes, traces_dir, "Read Sizes", "read_sizes", nbins=500)

    # Keep a backup of the original
    # Rename tmp to replace original
    os.rename(read_trace, f"{read_trace}.bk")
    os.rename(tmp_out, read_trace)