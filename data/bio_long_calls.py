import os
import argparse

# Latency is in ns so this would be a 1s disk operation
LATENCY_THRESHOLD = 1_000_000_000

def process_trace(bio_trace, lat_threshold):

    tracefile = open(bio_trace, "r")

    tmp_out = bio_trace + "_tmp"

    outfile = open(tmp_out, "w")

    for i, line in enumerate(tracefile):
        cols = " ".join(line.split()).split(" ")

        # latency = int(cols[6]) # For new bio trace
        latency = int(cols[8]) # old bio trace


        if latency > lat_threshold:
            print(f"Line {i}: Removing long latency operation of {latency:,} ns")
        else:
            outfile.write(line)

    tracefile.close()
    outfile.close()
    # Overwrite the original trace with the cleaned up version
    os.rename(tmp_out, bio_trace)

def print_long_calls(bio_trace, lat_threshold):

    tracefile = open(bio_trace, "r")

    for i, line in enumerate(tracefile):
        cols = " ".join(line.split()).split(" ")

        # latency = int(cols[6]) # For new bio trace
        latency = int(cols[8]) # old bio trace

        if latency > lat_threshold:
            print(f"{cols[0]} (line {i}): Long latency of {latency:,} ns")

    tracefile.close()


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Go through the bio trace and remove any long operations")
    p.add_argument("bio_trace", help="Time aligned bio trace")
    p.add_argument("-p", "--just-print", action='store_true', help="Only print out the long calls")
    args = p.parse_args()

    if not os.path.isfile(args.bio_trace):
        print(f"ERROR: Invalid bio trace {args.bio_trace}")
        exit(-1) 

    print(f"Checking bio trace for operations longer than {LATENCY_THRESHOLD:,} ns")

    if args.just_print:
        print_long_calls(args.bio_trace, LATENCY_THRESHOLD)
    else:
        process_trace(args.bio_trace, LATENCY_THRESHOLD)

    print("All done\n")