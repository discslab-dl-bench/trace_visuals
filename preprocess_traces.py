import pathlib
import argparse
from os import path

from preproc.mllog import process_mllog
from preproc.gpu import process_gpu_trace
from preproc.cpu import process_cpu_trace
from preproc.align_time import convert_traces_timestamp_to_UTC
from preproc.bio import filter_out_unwanted_processes_from_bio, process_long_bio_calls
from preproc.pids import get_pids
from preproc.iostat import iostat_to_csv
from preproc.read import process_absurdly_large_read_sizes
from preproc.utilities import iostat_trace_is_present
from preproc.traces import prepare_traces_for_timeline_plot

# Depends on machine local time setting
# For discslab-server2, UTC-4 during Daylight Savings Time and UTC-5 otherwise
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

# Change this if the traces change
TRACE_LATENCY_COLUMN_IDX = {
    'bio': 7,   
    'openat': 3,
    'read': 4,
    'write': 4,
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


def preprocess_traces(traces_dir, preproc_traces_dir, workload, skip_to=0):
    """
    Preprocessing pipeline.
    First, process the application log
    """
    # Process the MLLOG first
    if workload == 'dlio':
        print(f'DLIO log processing not implemented yet')
    
    if skip_to < 1:
        process_mllog(traces_dir, output_dir, workload)

    if skip_to < 2:
        # Time-align traces
        # Cut out everything captured before the MLLOG initialization event
        convert_traces_timestamp_to_UTC(traces_dir, preproc_traces_dir, TRACES, TRACES_AND_EXPECTED_NUM_COLUMNS, UTC_TIME_DELTA)
    
    # Some extra preprocessing
    if skip_to < 3:
        # Remove p99 latency bio calls - for plotting ~aesthetics~
        process_long_bio_calls(preproc_traces_dir)
        # Remove unwanted processes form the bio trace
        filter_out_unwanted_processes_from_bio(preproc_traces_dir, workload)
        # We sometimes get some absurdly large (i.e. > 10^20 B) returned read byes in the read trace
        # during initialization for unet3d when it's reading /sys/class/net/eth0/speed
        # we can consider this outliers and remove them
        process_absurdly_large_read_sizes(preproc_traces_dir)

    # Get the PIDs
    parent_pids, dataloader_pids, ignore_pids = get_pids(traces_dir, preproc_traces_dir)

    if skip_to < 4:
        prepare_traces_for_timeline_plot(preproc_traces_dir, parent_pids, dataloader_pids, ignore_pids, TRACES, TRACE_LATENCY_COLUMN_IDX)

    if skip_to < 5:
        # Use the ignore PIDs to preprocess the GPU trace
        process_gpu_trace(traces_dir, preproc_traces_dir, ignore_pids, UTC_TIME_DELTA)

        # CPU trace is system-wide so no PID filtering needed
        process_cpu_trace(traces_dir, preproc_traces_dir, UTC_TIME_DELTA)
        # Iostat trace as well, and is optional
        if iostat_trace_is_present(traces_dir):
            iostat_to_csv(traces_dir, preproc_traces_dir, UTC_TIME_DELTA)



if __name__=='__main__':
    p = argparse.ArgumentParser(description="Preprocess the output of the tracing scripts for plotting")
    p.add_argument("traces_dir", help="Directory where raw traces are")
    p.add_argument("workload", help="Which workload was run", choices=['unet3d', 'bert', 'dlrm', 'dlio'])
    p.add_argument("-o", "--output-dir", default="data_processed", help="Processed traces directory. Default is 'data_processed/'")
    p.add_argument("-s", "--skip-to", type=int, default=0, help="Skip to a certain step in the pipeline.")
    args = p.parse_args()

    traces_dir = args.traces_dir
    trace_basename = path.basename(args.traces_dir)
    output_dir = path.join(args.output_dir, trace_basename)
    workload = args.workload
    skip_to = args.skip_to

    if not path.isdir(traces_dir):
        print(f"ERROR: Invalid trace directory {traces_dir}")
        exit(-1) 

    # In the output directory, create a subdir with the same name as the input directory
    if not path.isdir(output_dir):
        print(f"Creating output directory {output_dir}")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not verify_all_traces_present(traces_dir, workload):
        exit(1)

    preprocess_traces(traces_dir, output_dir, workload, skip_to)




