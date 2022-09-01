import os
import re
import json
import argparse
import pathlib

def get_fields(line):
    return " ".join(line.split()).split(" ")


def main(data_dir, output_dir):

    pids_trace = os.path.join(data_dir, "pids.out")
    pids_trace = open(pids_trace, 'r')

    pid_names = {}

    for line in pids_trace:
        fields = get_fields(line)
        if fields[0] != "root":
            continue
        # Main process
        if re.match(r".*run_pretraining.py.*",line):
            if fields[1] == fields[2]:
                pid_names[fields[1]] = f"workers"
        else:
            continue
    
    print(f"Extracted PID information:\n{json.dumps(pid_names, indent=2)}\n")

    outfile = os.path.join(output_dir, "pids.json")
    outfile = open(outfile, 'w')
    json.dump(pid_names, outfile, indent=4)

    justpidsfile = os.path.join(output_dir, "pids")
    justpidsfile = open(justpidsfile, 'w')
    
    for pid in pid_names.keys():
        justpidsfile.write(f"{pid}\n")
      

if __name__ == "__main__":

    print('#####################################################')
    print("pid_names.py: Extracting PID information from traces")
    print('#####################################################\n')

    p = argparse.ArgumentParser(description="Extract relevant PIDs and their names from pids_tids.out")
    p.add_argument("data_dir", help="Raw traces directory")
    p.add_argument("output_dir", help="output directory")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Invalid data dir given")
        exit(-1) 
    
    if not os.path.isdir(args.data_dir):
        pathlib.Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    
    main(args.data_dir, args.output_dir)

    print("All done\n")
