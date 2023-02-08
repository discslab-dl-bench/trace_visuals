import os
import re
import json
import argparse
import pathlib

from os import path
from .utilities import get_fields


def get_pids(raw_traces_dir, preproc_traces_dir):
    """
    Determine the workload PIDs from processed read and raw GPU trace.
    Returns the parent PIDs, data-loader PIDs if present, and PIDs to ingore.
    """
    gpu_trace = path.join(raw_traces_dir, "gpu.out")
    read_trace = path.join(preproc_traces_dir, "read.out")

    pids_gpu, ignore_pids = get_pids_from_raw_gpu_trace(gpu_trace)
    pids_read = get_pids_from_read_trace(read_trace)

    # We usually filter the read trace by process name, with
    # an appropriate name for each workload. So the PIDs present
    # in it should all be related to the workload.
    #
    # The GPU trace however, could contain other unrelated PIDs if
    # another process was using the GPUs at the same time.
    # We potentially obtained some PIDs to ignore after parsing the GPU
    # trace.
    # 
    # Additionally, in the presence of PyTorch data-loading workers,
    # the read trace will contain many more PIDs vs the GPU trace,
    # which will show only the GPU-bound PIDs, parents of the data-loaders.
    #
    # We want to be able to distinguish the parent processes from the workers
    # and filter out any unrelated processes from the GPU trace
    #
    # Given all of this, let S_READ be the set of PIDs extracted from the 
    # read trace, S_GPU the set of PIDs extracted from the GPU trace, and
    # S_IGNORE the set of PIDs to ignore determined by the GPU trace.
    #
    # We will say that:
    #       S_GPU intersect S_READ = parent processes
    #       S_READ \ S_GPU = all data-loading workers
    #       S_IGNORE + (S_GPU \ S_READ) = processes to ignore            

    parent_pids = pids_gpu.intersection(pids_read)
    dataloader_pids = pids_read.difference(pids_gpu)
    ignore_pids = ignore_pids.union(pids_gpu.difference(pids_read))

    print(f'Parent PIDs ({len(parent_pids)}):\n{parent_pids}')
    print(f'Dataloader PIDs ({len(dataloader_pids)}):\n{dataloader_pids}')
    print(f'Ignore PIDs ({len(ignore_pids)}):\n{ignore_pids}')
    
    return parent_pids, dataloader_pids, ignore_pids


def get_pids_from_read_trace(read_trace):
    pids = set()
    with open(read_trace, 'r') as trace:
        for line in trace:
            data = get_fields(line)
            pid = data[1]

            if pid not in pids:
                pids.add(pid)
    print(f'Found {len(pids)} unique PIDs in the read trace.')

    return pids


def get_pids_from_raw_gpu_trace(gpu_trace):
    """
    Extract PIDs from GPU trace.
    Will skip the later PID if multiple are found running on the same GPU.
    """
    # Line format is 
    # 20230118   09:47:05      4    2647252     C     0     0     -     -   308   python 
    pat = re.compile(r'^\s+\d{8}\s+\d{2}:\d{2}:\d{2}\s+(\d+)\s+(\d+)')
    
    ignore_pids = set()

    # We map each GPU to a PID
    # We don't allow a GPU to have multiple PIDs (would indicate multiple workloads running at the same time)
    # but we could have a single PID using multiple GPUs e.g. DLRM or BERT with MirroredDistributionStrategy
    gpu_to_pid = {}

    last_line = ""

    with open(gpu_trace, 'r') as gpu_trace:
        for line in gpu_trace:
            if match := pat.match(line):
                gpu = match.group(1)
                pid = match.group(2)

                if gpu in gpu_to_pid:
                    if pid != gpu_to_pid[gpu]:
                        print(f"Identified multiple processes on GPU {gpu}:\n{last_line}{line}")
                        ignore_pids.add(pid)
                else:
                    gpu_to_pid[gpu] = pid
                
                last_line = line

    pids = set(gpu_to_pid.values())
    print(f"Found {len(pids)} GPU-bound PIDs, {len(ignore_pids)} PID{'s' if len(ignore_pids) > 1 else ''} to ingore in the GPU trace.")
    return pids, ignore_pids


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
        if re.match(r'.*launch.py.*', line):
            run_method = "launch.py"
            break
        elif re.match(r'.*resource_tracker.*', line):
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
                # We ignore the master process here since in practice it does not do much 
                # and takes up space in the timeline plots for nothing!
                continue
                # fields = get_fields(line)
                # pid_names[fields[1]] = "master"

            elif re.match(r".*\-u main\.py.*", line):
                fields = get_fields(line)
                # Keep only the line for the parent process
                # i.e. the one whose thread id is equal to the pid
                # (each process has many threads spawned)
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