import os
import re

from .utilities import get_write_trace, plot_histogram

# Column index of latency in the bio trace.
# Latest columns are
# TIMESTAMP         PID      OFFSET   SIZE (B)   LAT (ns)   FILENAME
WRITE_SZ_IDX = 3
FILENAME_IDX = 5

LINE_REGEX = re.compile(r'^\d{4}\-\d{2}\-\d{2}T\d{2}\:\d{2}\:\d{2}\.\d{9}\s+\d+\s+\w+\s+(\d+)\s+(\d+)\s+([\w\[\]\.\-]*)$')
REGEX_LOGGING_WRITE = re.compile(r'0|.*\.log')

def remove_logging_writes(traces_dir, workload) -> None:
    """
    We sometimes get some absurdly long returned read sizes in the read trace
    when reading /sys/class/net/eth0/speed. Remove them.
    TIMESTAMP         PID      OFFSET   RET (B)               LAT(ns)      FILENAME
    134078025034623   1797499  0x0      18446744073709551103  14865        speed
    134078025170518   1797505  0x0      18446744073709551103  14637        speed
    """
    print('Removing all writes related to logging')

    write_trace = get_write_trace(traces_dir)
    tmp_out = write_trace + "_tmp"

    all_write_sizes = []
    all_write_latencies = []
    lines_removed = 0

    with open(write_trace, "r") as tracefile, open(tmp_out, "w") as outfile:
        for i, line in enumerate(tracefile):

            if match := re.match(LINE_REGEX, line):
                size = int(match.group(1))
                lat = int(match.group(2))
                filename = match.group(3)
                
                if re.match(REGEX_LOGGING_WRITE, filename):
                    lines_removed += 1
                else:
                    outfile.write(line)
                    all_write_sizes.append(size)
                    all_write_latencies.append(lat)
            else:
                print(f'Could not match line {line}')

    print(f'\tRemoved {lines_removed} logging writes!')
    try:
        plot_histogram(all_write_sizes, traces_dir, "Write Sizes", "write_sizes", nbins=500)
        plot_histogram(all_write_latencies, traces_dir, "Write Latencies", "write_latencies", nbins=500)
    except:
        pass

    # Keep a backup of the original
    # Rename tmp to replace original
    os.rename(write_trace, f"{write_trace}.bk")
    os.rename(tmp_out, write_trace)