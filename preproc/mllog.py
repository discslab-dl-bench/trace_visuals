import os
import re
import json
import argparse
import pathlib
from pathlib import Path

import numpy as np
from preproc.utilities import _get_canonical_event_name

# Might have to modify this for DLIO
# Workloads have different events of interst based on their inner workings
EVENTS_OF_INTEREST = {
    'unet3d': {"init_start","init_stop","epoch_start","epoch_stop","eval_start","eval_stop","checkpoint_start","checkpoint_stop","ALL CASES SEEN"},
    'dlrm': {"init_start","init_stop","eval_start","eval_stop","training_start","training_stop","checkpoint_start","checkpoint_stop"},
    'bert': {"init_start","init_stop","block_start","block_stop","checkpoint_start","checkpoint_stop","eval_start","eval_stop"},
    'dlio': {"init_start","init_stop","block_start","block_stop","eval_start","eval_stop","training_start","training_stop","checkpoint_start","checkpoint_stop"},
}

MLLOG_LINE_REGEX = r'(\[[0-9]\]\s){0,1}:::MLLOG\s'

def process_mllog(traces_dir, output_dir, workload):
    mllog_to_valid_json(traces_dir, output_dir, workload)
    create_timeline_csv(output_dir, workload)


def get_init_start_time(preproc_traces_dir):
    """
    Returns the timestamp of the MLLOG INIT_START event.
    """
    timeline_csv = os.path.join(preproc_traces_dir, "timeline", "timeline.csv")
    with open(timeline_csv, 'r') as timeline:
        for line in timeline:
            data = line.replace("\n", "").split(",")
            if data[2] == "INIT":
                return np.datetime64(data[0])

    raise Exception(f'ERROR: Could not find INIT event in {timeline_csv}')


def get_workload_app_log(traces_dir, workload):
                
    possible_log_file_names = {
         'bert': ['app.log', 'bert.log'],
         'dlrm': ['dlrm.log', 'dlrm_tera.log'],
         'dlio': ['dlio.log'],
         'unet3d': ['unet3d.log'],
    }

    for potential_log_file in possible_log_file_names[workload]:

        matching_files = list(Path(traces_dir).rglob(potential_log_file))

        if len(matching_files) > 0:
            return str(matching_files[0])
    
    raise Exception(f'ERROR: Could not find application log for {workload}')


def mllog_to_valid_json(traces_dir, output_dir, workload):
    """
    Go through mllog, transform to valid JSON and filter events based on workload.
    """
    print('Converting mllog to valid JSON')
    logfile = get_workload_app_log(traces_dir, workload)
    outfile = os.path.join(output_dir, f'{workload}.log')

    p_mllog_line = re.compile(MLLOG_LINE_REGEX)

    with open(logfile, 'r') as log, open(outfile, 'w') as outfile:
        # Open a JSON array
        outfile.write('[\n')
        for line in log:
            if match := re.match(p_mllog_line, line):
                
                if (horovod_proc := match.group(1)) is not None:
                    horovod_proc = horovod_proc.strip()
                    if horovod_proc != '[0]':
                        continue

                line = line.replace(match.group(0), '').strip()
                # Convert the UNIX timestamp to UTC before writing back
                data = json.loads(line)
                data['time_ms'] = str(np.datetime64(data["time_ms"], "ms"))
                line = json.dumps(data)

                line += ',\n'
                outfile.write(line)

        # Remove trailing comma and newline by going back 2 characters from
        # the current file position given by outfile.tell() and truncating
        outfile.seek(outfile.tell() - 2)
        outfile.truncate()
        # End the JSON array
        outfile.write('\n]\n')



def create_timeline_csv(preprocessed_traces_dir, workload):
    """
    Convert the UNIX timestamps of the mllog to UTC timestamp.
    """
    print('Creating a timeline.csv from mllog')
    preproc_log = os.path.join(preprocessed_traces_dir, f"{workload}.log")

    outdir = os.path.join(preprocessed_traces_dir, "timeline")
    if not os.path.isdir(outdir):
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    output_csv = os.path.join(outdir, "timeline.csv")

    events_of_interest = EVENTS_OF_INTEREST[workload]

    with open(preproc_log, 'r') as infile, open(output_csv, 'w') as outfile:
        all_logs = json.load(infile)

        started_events = {}

        for log in all_logs:

            if "key" in log and log['key'] in events_of_interest:
                timestamp = log["time_ms"]

                key_parts = log["key"].split("_")
                if len(key_parts) < 2:
                    continue

                evt = _get_canonical_event_name(key_parts[0].upper())
                evt_type = key_parts[1].upper()

                if evt_type == "STOP":
                    if evt not in started_events:
                        print(f"WARNING: No starting event for {log['key']} at ts {log['time_ms']}\n")
                        continue
                    else:
                        outfile.write(f"{started_events[evt]},{timestamp},{evt}\n")
                        del started_events[evt]
                else:
                    started_events[evt] = timestamp


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Changes the UNIX timestamp in the mllog to a UTC timestamp")
    p.add_argument("traces_dir", help="Directory where raw mllog is found")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("workload", choices=['unet3d', 'dlrm', 'bert', 'dlio'], help="Workload name")
    args = p.parse_args()

    if not os.path.isdir(args.traces_dir):
        print(f"Invalid traces_dir given")
        exit(-1)

    if not os.path.isdir(args.output_dir):
        print(f"Creating output directory {args.output_dir}")
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    mllog_to_valid_json(args.traces_dir, args.output_dir, args.workload)
    create_timeline_csv(args.output_dir)
