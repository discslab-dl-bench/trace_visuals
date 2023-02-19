import os
import glob
import argparse
import re
import pathlib

from .utilities import _get_canonical_event_name, get_dlio_log

DLIO_LOG_REGEX = r'\[[A-Z]+\]\s([0-9]{4}\-[0-9]{2}\-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6})\s([\w\s]+)'

DLIO_EVENT_REGEX = r'(Starting|Ending)\s+(block|eval|checkpoint)'

def process_dlio_log(data_dir, output_dir):
    """
    DLIO logs are already in UTC
    """

    dlio_log = get_dlio_log(data_dir)
    outdir = os.path.join(output_dir, "timeline")
    if not os.path.isdir(outdir):
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    output_csv = os.path.join(outdir, "timeline.csv")

    p_dlio_log_line = re.compile(DLIO_LOG_REGEX)
    p_dlio_event = re.compile(DLIO_EVENT_REGEX)

    with open(dlio_log, 'r') as dlio_log, open(output_csv, 'w') as outfile:

        # Because DLIO does not log an initialization event,
        # we consider everything until the opening of the first
        # training block to be initialization.
        first_line = next(dlio_log)
        match = re.match(p_dlio_log_line, first_line)
        init_start_ts = match.group(1)
        
        init_stoppped = False

        dlio_log.seek(0)
        started_events = {}

        for line in dlio_log:

            if match := re.match(p_dlio_log_line, line):
                ts = match.group(1)
                line_text = match.group(2)

                if match := re.match(p_dlio_event, line_text):
                    start_stop = match.group(1)
                    event = match.group(2).upper()
                    event = _get_canonical_event_name(event)

                    if start_stop == "Ending":
                        if event not in started_events:
                            print(f"WARNING: No starting event for {event} at {ts}\n")
                            continue
                        else:
                            if not init_stoppped:
                                outfile.write(f"{init_start_ts},{started_events[event]},INIT\n")
                                init_stoppped = True
                            outfile.write(f"{started_events[event]},{ts},{event}\n")
                            del started_events[event]
                    else:
                        started_events[event] = ts

            else:
                print(f'Could not match line {line}')



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