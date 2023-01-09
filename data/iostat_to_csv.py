import os
import json
import argparse
import pathlib
import numpy as np

from datetime import datetime

# We are in eastern time, so UTC-4 during DST, UTC-5 otherwise
UTC_TIME_DELTA = 5

def iostat_to_csv(iostat_log, outdir):
    """
    Convert the timestamp of the iostat log to UTC
    Write out a csv containing the recorded metrics for each disk
    """

    with open(iostat_log, "r") as infile:
        log = json.load(infile)

    with open(f"{outdir}/iostat.csv", "w") as outcsv:
        
        # Write the header
        first_diskstat = log["sysstat"]["hosts"][0]["statistics"][0]["disk"][0]
                
        headers = ["timestamp"]

        for header in first_diskstat.keys():
            headers.append(header)

        print(headers)
        outcsv.write(",".join(headers) + "\n")

        # Iterate through the iostat file and write the values in CSV format
        for host in log["sysstat"]["hosts"]:
            for statline in host["statistics"]:
                timestamp = statline["timestamp"]

                date, time, am_pm = timestamp.split(" ")
                
                # Convert the AM/PM time to a 24h time
                time = datetime.strptime(f"{time} {am_pm}", "%I:%M:%S %p")
                time = datetime.strftime(time, "%H:%M:%S")

                timestamp = str(np.datetime64(f'{date}T{time}') + np.timedelta64(UTC_TIME_DELTA, "h"))

                # Contains an data object for each disk, we write out one line per disk
                for diskstats in statline["disk"]:
                    cols = [timestamp]

                    for value in diskstats.values():
                        cols.append(str(value))

                    outcsv.write(",".join(cols) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Changes the timestamps in the iostat trace to UTC")
    p.add_argument("iostat_log", help="iostat log")
    p.add_argument("outdir", help="Output directory")
    args = p.parse_args()

    if not os.path.isfile(args.iostat_log):
        print(f"Invalid iostat_log given")
        exit(-1) 

    if not os.path.isdir(args.outdir):
        print(f"Output dir does not exist. Creating.")
        pathlib.Path(args.data_dir).mkdir(exist_ok=True, parents=True)

    iostat_to_csv(args.iostat_log, args.outdir)
