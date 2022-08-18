import os
import json
import argparse
import numpy as np

def process_timeline(datadir):
    """
    Convert the UNIX timestamps of the mllog to UTC timestamp
    """
    infile = open(f"{datadir}/timeline.log", "r")
    outfile = open(f"{datadir}/timeline.csv", "w")

    all_logs = json.load(infile)

    started_events = {}
    have_not_seen_epoch = True

    for i, log in enumerate(all_logs):

        ux_time = np.datetime64(log["time_ms"], "ms") + np.timedelta64(5, "h")
        # ux_time = np.datetime64(log["time_ms"], "ms")
        key_parts = log["key"].split("_")

        evt = key_parts[0].upper()
        evt_type = key_parts[1]

        if evt_type == "stop":
            if evt not in started_events:
                print(f"No starting event for {log['key']} at ts {log['time_ms']}\n")
                continue
            else:
                # Label the first epoch differently
                if evt == "EPOCH" and have_not_seen_epoch:
                    outfile.write(f"{started_events[evt]},{ux_time},EPOCH\n")
                    have_not_seen_epoch = False
                else:
                    outfile.write(f"{started_events[evt]},{ux_time},{evt}\n")
                del started_events[evt]
        else:
            started_events[evt] = ux_time


def process_vals():

    cwd = os.path.basename(os.getcwd())
    if cwd == "data":
        path = "mllog_data"
    else:
        path = "data/mllog_data"

    infile = open(f"{path}/evals.log", "r")
    outfile = open(f"{path}/evals.csv", "w")

    logs = json.load(infile)

    for log in logs:

        ux_time = np.datetime64(log["time_ms"], "ms")
        score = log["value"]

        outfile.write(f"{ux_time},{score}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Changes the UNIX timestamp in the mllog to a UTC timestamp")
    p.add_argument("datadir", help="directory where processed unet3d.log is found")
    args = p.parse_args()

    if not os.path.isdir(args.datadir):
        print(f"Invalid data directory given")
        exit(-1) 

    process_timeline(args.datadir)
    # process_vals()
