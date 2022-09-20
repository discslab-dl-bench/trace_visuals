import os
import re
import json
import argparse
import pathlib

def get_fields(line):
    return " ".join(line.split()).split(" ")


def main(data_dir, output_dir):

    pids_trace = os.path.join(data_dir, "pids_tids.out")

    if not os.path.isfile(pids_trace):
        print(f"pids_tids.out not found. Looking for pids.out")

    pids_trace = os.path.join(data_dir, "pids.out")

    if not os.path.isfile(pids_trace):
        print(f"pids.out not found! Failed.")
        exit()
        
    pids_trace = open(pids_trace, 'r')

    # Identify the run method
    run_method = None
    for line in pids_trace:
        if re.match(r'.*resource_tracker.*', line):
            run_method = "mp.spawn"
            break
        
    pids_trace.seek(0)
    for line in pids_trace:
        if re.findall(r"mpirun",line):
            run_method = "mpirun"
            break
    

    if run_method is None:
        run_method = "launch.py"

    print(f"Found evidence that the run_method was {run_method}.\nExtracting PID info appropriately.")
    pid_names = {}

    pids_trace.seek(0)
    # Case 1: we used my mp.spawn to launch training.
    # In this case, 
    if run_method == "mp.spawn":
        num_worker = 1
        for line in pids_trace:
            # Main process
            if re.match(r".*python main\.py.*", line):
                fields = get_fields(line)
                pid_names[fields[1]] = "master"
            
            elif re.match(r".*resource_tracker.*", line):
                fields = get_fields(line)
                pid_names[fields[1]] = "resource tracker"
            
            elif re.match(r".*spawn.*", line):
                fields = get_fields(line)
                # Keep only the PIDs (we originally printed the TIDs as well)
                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"worker {num_worker}"
                    num_worker += 1
            else:
                continue

    elif run_method == "mpirun":
        num_worker = 1
        for line in pids_trace:
            if re.findall(r"/bin/dash -c mpirun",line):
                fields = get_fields(line)
                pid_names[fields[1]] = "master"
            elif re.findall(r"src/dlio_benchmark.py",line):
                fields = get_fields(line)
                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"worker {num_worker}"
                    num_worker += 1
                else:
                    continue
    # Case 3: we used launch.py to launch training.
    # In this case, all relevant lines will include 'main.py'
    else:
        num_worker = 1
        for line in pids_trace:
            if re.match(r".*launch\.py.*", line):
                fields = get_fields(line)
                pid_names[fields[1]] = "master"

            elif re.match(r".*\-u main\.py.*", line):
                fields = get_fields(line)

                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"worker {num_worker}"
                    num_worker += 1
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