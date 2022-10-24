import os
import glob
import argparse
import re


def find_log(data_dir):
    '''
    Find the log file name with correct info needed
    '''
    all_log_files = glob.glob(f'{data_dir}/*.log')
    
    for log_file in all_log_files:
        with open(log_file,"r") as f:
            for l in f.readlines():
                train_log = re.findall(r"Starting epoch",l)
                if train_log:
                    return log_file

def generate_csv(log_file_name, output_dir):
    '''
    Generate timeline.csv file for plotting
    '''

    outfile = open(f"{output_dir}/timeline.csv", "w")
    line_elements = ["", "", ""]

    with open(log_file_name, "r") as f:
        for l in f.readlines():
            train_start = re.findall(r"Starting epoch",l)
            train_end = re.findall(r"Ending epoch",l)

            eval_start = re.findall(r"Starting eval",l)
            eval_end = re.findall(r"Ending eval",l)
            utc_time = l.split(' ')[0]
            if train_start:
                line_elements[0] = utc_time
                line_elements[2] = "EPOCH"
            if eval_start:
                line_elements[0] = utc_time
                line_elements[2] = "EVAL"

            if train_end or eval_end:
                line_elements[1] = utc_time
                line_to_write = ",".join(line_elements)
                line_to_write = f"{line_to_write}\n"
                outfile.write(line_to_write)
                line_elements = ["", "", ""]



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess DLIO log files for plotting")
    parser.add_argument("data_dir", help="Raw traces directory")
    parser.add_argument("output_dir", help="output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    log_file_name = find_log(args.data_dir)
    generate_csv(log_file_name, args.output_dir)