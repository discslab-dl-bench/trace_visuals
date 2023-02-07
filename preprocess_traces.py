import pathlib
import argparse
from os import path

from preproc.mllog import process_mllog
from preproc.align_time import convert_traces_timestamp_to_UTC
from preproc.bio import process_long_bio_calls
from preproc.pids import get_pids

# We are in eastern time, so UTC-4 during DST and UTC-5 otherwise
UTC_TIME_DELTA = 5

# Add any new trace we want to time align here
# We don't care about close, create_del for plotting (for now)
TRACES = ["bio", "openat", "read", "write"]

# Some lines can have an extra column or two, e.g. if reading from a file with a name vs anonymous
TRACES_AND_EXPECTED_NUM_COLUMNS = {
    "bio": [8, 9],      # Old bio trace was 9, new one is 8
    "openat": [5, 6],   # Both 5 and 6 are valid
    "read": [5, 6],
    "write": [5, 6]     # Old trace was [8, 9]    
}


def verify_all_traces_present(traces_dir, workload) -> bool:
    """
    Returns true if all essential traces are present with their expected file name.
    """
    all_traces = [f'trace_{trace}.out' for trace in TRACES] 
    all_traces += ['gpu.out', 'cpu.out', f'{workload}.log', 'trace_time_align.out']

    success = True
    for trace in all_traces:
        expected_filename = path.join(traces_dir, trace)

        if not path.isfile(expected_filename):
            print(f'ERROR: Missing essential trace {expected_filename}')
            success = False
        
    return success


def preprocess_traces(traces_dir, preproc_traces_dir, workload):
    """
    Preprocessing pipeline.
    First, process the application log
    """
    # Process the MLLOG first
    # Cut out everything before initialization before time-aligning traces
    process_mllog(traces_dir, output_dir, workload)

    # Time-align traces
    convert_traces_timestamp_to_UTC(traces_dir, preproc_traces_dir, TRACES, TRACES_AND_EXPECTED_NUM_COLUMNS, UTC_TIME_DELTA)

    # Remove p99 latency bio calls
    bio_trace_UTC = path.join(preproc_traces_dir, "bio_UTC.out")
    process_long_bio_calls(bio_trace_UTC, preproc_traces_dir, verbose=True)

    # Get the PIDs
    parent_pids, dataloader_pids, ignore_pids = get_pids(traces_dir, preproc_traces_dir)

    # Use the ignore PIDs to preprocess the GPU trace




if __name__=='__main__':
    p = argparse.ArgumentParser(description="Preprocess the output of the tracing scripts for plotting")
    p.add_argument("traces_dir", help="Directory where raw traces are")
    p.add_argument("workload", help="Which workload was run", choices=['unet3d', 'bert', 'dlrm', 'dlio'])
    p.add_argument("-o", "--output-dir", default="data_processed", help="Directory where to write the time aligned traces")
    args = p.parse_args()

    traces_dir = args.traces_dir
    trace_basename = path.basename(args.traces_dir)
    output_dir = path.join(args.output_dir, trace_basename)
    workload = args.workload

    if not path.isdir(traces_dir):
        print(f"ERROR: Invalid trace directory {traces_dir}")
        exit(-1) 

    # In the output directory, create a subdir with the same name as the input directory
    if not path.isdir(output_dir):
        print(f"Creating output directory {output_dir}")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not verify_all_traces_present(traces_dir, workload):
        exit(1)

    preprocess_traces(traces_dir, output_dir, workload)




