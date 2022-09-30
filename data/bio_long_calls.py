import os
import argparse

# Latency is in ns so this would be a 100ms disk operation
LATENCY_THRESHOLD = 100_000_000_000

def process_trace(bio_trace, lat_threshold):

    tracefile = open(bio_trace, "r")

    tmp_out = bio_trace + "_tmp"

    outfile = open(tmp_out, "w")

    for i, line in enumerate(tracefile):
        cols = " ".join(line.split()).split(" ")

        latency = int(cols[7])

        if latency > lat_threshold:
            print(f"Line {i}: Removing long latency operation of {latency:,} ns")
        else:
            outfile.write(line)

    tracefile.close()
    outfile.close()
    # Overwrite the original trace with the cleaned up version
    os.rename(tmp_out, bio_trace)



if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Go through the bio trace and remove any long operations")
    p.add_argument("bio_trace", help="Time aligned bio trace")
    args = p.parse_args()

    if not os.path.isfile(args.bio_trace):
        print(f"ERROR: Invalid bio trace {args.bio_trace}")
        exit(-1) 

    print(f"Checking bio trace for operations longer than {LATENCY_THRESHOLD:,} ms")

    process_trace(args.bio_trace, LATENCY_THRESHOLD)

    print("All done\n")