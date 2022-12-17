import os
import argparse

# Latency is in ns so this would be a 1s disk operation
LATENCY_THRESHOLD = 1_000_000_000

# 6 for new traces, 8 for old
# Set to 8 in main() if the old-trace flag is passed
COLUMN_INDEX = 6

def process_trace(bio_trace, lat_threshold, just_print=True):

    tracefile = open(bio_trace, "r")

    if not just_print:
        tmp_out = bio_trace + "_tmp"
        outfile = open(tmp_out, "w")

    for i, line in enumerate(tracefile):
        cols = " ".join(line.split()).split(" ")
        latency = int(cols[COLUMN_INDEX])

        if latency > lat_threshold:
            print(f"{cols[0]} (line {i}): Long latency of {latency:,} ns")
        else:
            if not just_print:
                outfile.write(line)

    tracefile.close()

    # Overwrite the original trace with the cleaned up version
    if not just_print:
        outfile.close()
        os.rename(tmp_out, bio_trace)

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Go through the bio trace and remove any long operations")
    p.add_argument("bio_trace", help="Time aligned bio trace")
    p.add_argument("-t", "--threshold", type=int, help="Detection threshold in ns (1s by default)")
    p.add_argument("-p", "--just-print", action='store_true', help="Only print out the long calls")
    p.add_argument("-o", "--old-trace", action='store_true', help="Process an old bio trace")
    args = p.parse_args()

    if not os.path.isfile(args.bio_trace):
        print(f"ERROR: Invalid bio trace {args.bio_trace}")
        exit(-1) 

    if args.threshold:
        threshold = args.threshold
    else:
        threshold = LATENCY_THRESHOLD

    if args.old_trace:
        COLUMN_INDEX = 8

    print(f"Checking bio trace for operations longer than {threshold:,} ns")

    process_trace(args.bio_trace, threshold, args.just_print)

    print("All done\n")