import os
import re
import json
import argparse
import pathlib

import numpy as np

# Might have to modify this for DLIO
# Workloads have different events of interst based on their inner workings
WORKLOAD_MLLOG_REGEX_PATTERN = {
    'unet3d': r".*(init_start|init_stop|epoch_start|epoch_stop|eval_start|eval_stop|checkpoint_start|checkpoint_stop).*",
    'dlrm': r".*(init_start|init_stop|block_start|block_stop|eval_start|eval_stop|training_start|training_stop|checkpoint_start|checkpoint_stop).*",
    'bert': r".*(init_start|init_stop|block_start|block_stop|checkpoint_start|checkpoint_stop).*",
    'dlio': r".*(init_start|init_stop|block_start|block_stop|eval_start|eval_stop|training_start|training_stop|checkpoint_start|checkpoint_stop).*",
}


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


def mllog_to_valid_json(traces_dir, output_dir, workload):
    """
    Go through mllog, transform to valid JSON and filter events based on workload.
    """
    logfile = os.path.join(traces_dir, f'{workload}.log')
    outfile = os.path.join(output_dir, f'{workload}.log')

    regex = WORKLOAD_MLLOG_REGEX_PATTERN[workload]
    pattern = re.compile(regex)

    with open(logfile, 'r') as log, open(outfile, 'w') as outfile:
        # Open a JSON array
        outfile.write('[\n')
        for line in log:
            if re.match(pattern, line):
                line = line.replace(":::MLLOG ", "").rstrip()

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

    preproc_log = os.path.join(preprocessed_traces_dir, f"{workload}.log")

    outdir = os.path.join(preprocessed_traces_dir, "timeline")
    if not os.path.isdir(outdir):
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    output_csv = os.path.join(outdir, "timeline.csv")

    with open(preproc_log, 'r') as infile, open(output_csv, 'w') as outfile:
        all_logs = json.load(infile)

        started_events = {}
        have_not_seen_epoch = True

        for i, log in enumerate(all_logs):

            timestamp = log["time_ms"]
            key_parts = log["key"].split("_")

            evt = key_parts[0].upper()
            evt_type = key_parts[1].upper()

            if evt_type == "STOP":
                if evt not in started_events:
                    print(f"WARNING: No starting event for {log['key']} at ts {log['time_ms']}\n")
                    continue
                else:
                    # Label the first epoch differently
                    if evt == "EPOCH" and have_not_seen_epoch:
                        outfile.write(f"{started_events[evt]},{timestamp},EPOCH\n")
                        have_not_seen_epoch = False
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
