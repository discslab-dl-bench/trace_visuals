import os
import glob
import argparse
import re


def find_log(data_dir):
    '''
    Find the log file name with correct info needed
    '''
    all_log_files = glob.glob(f'{data_dir}/*.log')
    print(f"Found .log files: {all_log_files}")
    for log_file in all_log_files:
        with open(log_file,"r") as f:
            for l in f.readlines():
                if re.findall(r"Starting epoch",l):
                    return log_file


def generate_csv(log_file_name, output_dir):
    '''
    Generate timeline.csv file for plotting from DLIO log
    '''

    outfile = open(f"{output_dir}/timeline.csv", "w")
    line_elements = ["", "", ""]

    with open(log_file_name, "r") as f:
        started_events = {}
        have_not_seen_epoch = True

        for i, line in enumerate(f):
            if re.findall(r'Starting', line):

                result = re.search(r'Starting\s([a-zA-Z]+)\s', line)
                evt = result.group(1).upper()

                if evt == "EPOCH":
                    continue
                
                utc_time = line.split(' ')[1]
                started_events[evt] = utc_time

            elif re.findall(r'Ending', line):

                result = re.search(r'Ending\s([a-zA-Z]+)\s', line)
                evt = result.group(1).upper()

                utc_time = line.split(' ')[1]
                if evt not in started_events:
                    print(f"No starting event for {evt} at ts {utc_time}\n")
                    continue
                else:
                    outfile.write(f"{started_events[evt]},{utc_time},{evt}\n")
                    print(f"{started_events[evt]},{utc_time},{evt}")
                    del started_events[evt]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess DLIO log files for plotting")
    parser.add_argument("data_dir", help="Raw traces directory")
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    log_file_name = find_log(args.data_dir)
    generate_csv(log_file_name, args.output_dir)