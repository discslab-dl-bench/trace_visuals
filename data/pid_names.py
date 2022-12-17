import os
import re
import json
import argparse
import pathlib

def get_fields(line):
    """
        Split the line on whitespace, join it on a single space then split it again.
        This makes it return nicely delimited tokens because the original number of
        spaces is variable.
    """
    return " ".join(line.split()).split(" ")


def main(data_dir, output_dir):

    pids_trace = os.path.join(data_dir, "pids_tids.out")

    if not os.path.isfile(pids_trace):
        print(f"pids_tids.out not found. Looking for pids.out")

    pids_trace = os.path.join(data_dir, "pids.out")

    if not os.path.isfile(pids_trace):
        print(f"pids.out not found! Looking for pids_<date>.out")
    
    # sort the pids_date.out files to make sure we get pids.json correctly
    # Will open the pid file with larger timestamp if there are multiple
    # which is what we want since often the first one doesn't have all processes ready
    all_files = sorted(os.listdir(data_dir))
    for f in all_files:
        if re.match(r"pids_[0-9]*",f):
            pids_trace = os.path.join(data_dir,f)

    if not os.path.isfile(pids_trace):
        print(f"pids_<date>.out not found! Abort")
        exit()
    
    print(f"Opening pid file {pids_trace}")
    pids_trace = open(pids_trace, 'r')

    # Identify the run method
    run_method = None
    for line in pids_trace:
        if re.match(r'.*resource_tracker.*', line):
            run_method = "mp.spawn"
            break
        elif re.findall(r"mpirun",line):
            run_method = "mpirun"
            break
        elif re.findall(r"dlrm_s_pytorch",line):
            run_method = "dlrm"
            break
        elif re.findall(r"run_pretraining",line):
            run_method = "bert"
            break
        else:
            continue
    
    # Default run method, though now with multiple workloads this is no longer valid
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
            if re.findall(r"mpirun",line):
                # We can actually ignore the master process as it does not do much
                continue
                # fields = get_fields(line)
                # pid_names[fields[1]] = "master"
            elif re.findall(r"src/dlio_benchmark.py",line):
                fields = get_fields(line)
                # Checks if the PID == SPID, which corresponds to the first thread of the group
                # There are mny threads (dif SPIDs) for each PID, we only keep track of the first
                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"worker {num_worker}"
                    num_worker += 1
                else:
                    continue
            # mpirun is also used under hood with horovod, 
            # which can be used to launch the various workloads.
            # This catches BERT launched with horovod
            elif re.findall(r"run_pretraining.py",line):
                fields = get_fields(line)
                # Checks if the PID == SPID, which corresponds to the first thread of the group
                # There are mny threads (dif SPIDs) for each PID, we only keep track of the first
                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"worker {num_worker}"
                    num_worker += 1
                else:
                    continue

    elif run_method == "dlrm":
        num_worker = 1
        for line in pids_trace:
            if re.findall(r"run_in_container.sh",line):
                fields = get_fields(line)
                pid_names[fields[1]] = "launch script"
            elif re.findall(r"dlrm_s_pytorch.py",line):
                fields = get_fields(line)
                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"worker {num_worker}"
                    num_worker += 1
                else:
                    continue

    elif run_method == "bert":
        num_worker = 1
        for line in pids_trace:
            # if re.findall(r"train_model.sh",line):
            #     fields = get_fields(line)
            #     pid_names[fields[1]] = "launch script"
            if re.findall(r"run_pretraining.py",line):
                fields = get_fields(line)
                if fields[1] == fields[2]:
                    pid_names[fields[1]] = f"Worker {num_worker}"
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


    p = argparse.ArgumentParser(description="Extract relevant PIDs and their names from pids_tids.out")
    p.add_argument("data_dir", help="Raw traces directory")
    p.add_argument("output_dir", help="output directory")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Invalid data dir given")
        exit(-1) 
    
    print('#####################################################')
    print("pid_names.py: Extracting PID information from traces")
    print('#####################################################\n')

    if not os.path.isdir(args.data_dir):
        pathlib.Path(args.data_dir).mkdir(exist_ok=True, parents=True)
    
    main(args.data_dir, args.output_dir)

    print("All done\n")