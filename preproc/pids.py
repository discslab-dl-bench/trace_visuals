import os
import re
import json
import argparse
import pathlib

from os import path
from .utilities import get_fields


def get_pids(raw_traces_dir, preproc_traces_dir, workload):
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
    # in it are MOSTLY related to the workload. However, since we 
    # filter on process name, any other process with that name (e.g. python)
    # will get caught in the traces, which happens for the vscode language server 
    # for example, or someone running a random script.
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

    print(f'Parent PIDs ({len(parent_pids)}):\n{sorted(list(parent_pids))}')
    print(f'Dataloader PIDs ({len(dataloader_pids)}):\n{sorted(list(dataloader_pids))}')
    print(f'Ignore PIDs ({len(ignore_pids)}):\n{sorted(list(ignore_pids))}')

    # If we have dataloading processes and it's not UNET3D, add them to the ignore set
    # Note that we won't be able to catch foreign processes in this case.
    if workload == 'bert' and len(dataloader_pids) > 0:
        print(f'Found extra PIDs in the read trace for {workload}.')
        print(f'Most likely due to another python process starting during tracing. Ignore.')
        ignore_pids = ignore_pids.union(dataloader_pids)
        dataloader_pids = []

    # DLIO will always have an empty parent_pids since it does not use GPUs
    if workload == 'dlio':
        parent_pids = dataloader_pids
        dataloader_pids = {}
    
    return parent_pids, dataloader_pids, ignore_pids


def get_pids_from_read_trace(read_trace):
    print('Extracting PIDs from read trace')
    
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
    Should catch differnt PIDs running alone on GPUs later in the trace though, like for BERT evaluation.
    """
    print('Extracting PIDs from GPU trace')
    # Line format is 
    # 20230118   09:47:05      (4)    (2647252)     C     0     0     -     -   308   python 
    pat = re.compile(r'^\s+\d{8}\s+\d{2}:\d{2}:\d{2}\s+(\d+)\s+(\d+|\-)')
    
    ignore_pids = set()

    # We map each GPU to a PID
    # We don't allow a GPU to have multiple PIDs (would indicate multiple workloads running at the same time)
    # but we could have a single PID using multiple GPUs e.g. DLRM or BERT with MirroredDistributionStrategy
    pids = set()

    last_line = ""

    prev_gpu_idx = -1

    with open(gpu_trace, 'r') as gpu_trace:
        for line in gpu_trace:
            if match := pat.match(line):
                gpu = match.group(1)
                pid = match.group(2)
                
                # Lines with no processes on GPU have '-' as PID
                if pid != '-':
                    # If the current line is for the same GPU as the previous line,
                    # it means two processes are running on the same GPU, which should not 
                    # happen for our workloads. It would indicate someone else running 
                    # a workload at the same time. Ignore the second line's process.
                    if gpu == prev_gpu_idx:
                        # Only add the PID to the ignore list, if we haven't already included it in the PIDs of interest
                        if pid not in pids:
                            print(f"Identified concurrent process on GPU {gpu}:\n{last_line}{line}")
                            ignore_pids.add(pid)
                    else:
                        if pid not in pids:
                            pids.add(pid)
                
                last_line = line
                prev_gpu_idx = gpu

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

    pids, ignore_pids = get_pids_from_raw_gpu_trace('test_data/gpu_w_multiple_procs')
    assert pids == {'2720663', '2720662', '2720661'}
    assert ignore_pids == {'3333333'}
    print("GPU from nvidia-smi trace test PASS")
