import re
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

from step_breakdown import get_batch_size, get_num_gpus, DLRM_BATCH_SIZE_STRINGS


READ_HEADER = ["timestamp", "pid", "offset", "size", "latency", "filename"]
WRITE_HEADER = ["timestamp", "pid", "offset", "size", "latency", "filename"]
BIO_HEADER = ["timestamp", "pid", "command", "disk", "type", "size", "sector", "latency"]
TIMELINE_HEADER = ["start_time", "end_time", "event"]

DATA_FILE_PATTERN = re.compile(r'part\-|case_|\.bin|eval_10k|tfrecord|img_')
CKPT_FILE_PATTERN = re.compile(r'tempstate|model\.ckpt|model\-|ckpt')

workload_regex_patterns = {
    'UNET3D': re.compile(r'.*UNET|UNET3D.*')
}


def filter_df_by_time(df: pd.DataFrame, key, copy=False, start=None, end=None):
    """
    Filters a pd.DataFrame between start and end time.
    Assumes df[key] has a Timestamp dtype.
    Will deep copy the DataFrame if copy==True
    otherwise df will be overwritten.
    """
    if copy:
        df = df.copy(deep=True)
    if start is not None:
        df = df[df[key] >= np.datetime64(start)]
    if end is not None:
        df = df[df[key] <= np.datetime64(end)]
    return df


def preprocess_data(data_dirs, output_dir, workload):

    plotting_data = {}
    error = False

    for data_dir in data_dirs:
        p_workload = workload_regex_patterns[workload]
        log_dirs = [dir for dir in list(data_dir.iterdir()) if dir.is_dir() and re.match(p_workload, dir.name) ]
        print(log_dirs)

        for log_dir in sorted(log_dirs):
            exp_name = log_dir.name
            gpu_key = get_num_gpus(exp_name)
            batch_key = get_batch_size(exp_name)
            print(f'Processing {workload} - GPUs: {gpu_key} Batch Size: {batch_key}')

            if gpu_key not in plotting_data:
                plotting_data[gpu_key] = {
                    batch_key: { }
                }
            else:
                if batch_key not in plotting_data[gpu_key]: 
                    plotting_data[gpu_key][batch_key] = { }

            timeline = log_dir / 'processed/timeline/timeline.csv'
            timeline = pd.read_csv(timeline, on_bad_lines='skip', names=TIMELINE_HEADER, header=None, engine='c', delimiter=',', low_memory=False, parse_dates=['start_time', 'end_time'])

            # import code
            # code.interact(local=locals())

            if workload == 'UNET3D':
                fourth_epoch_start = timeline[timeline['event'] == 'TRAINING'].iloc[3].start_time


            read_trace = log_dir / "processed/read.out"
            if read_trace.exists():
                rdf = pd.read_csv(read_trace, on_bad_lines='skip', names=READ_HEADER, header=None, engine='c', delim_whitespace=True, low_memory=False, dtype={"size": float}, parse_dates=['timestamp'])
                rdf = rdf[rdf['filename'].str.contains(DATA_FILE_PATTERN, regex=True, na=False)]

                # Filter out some epochs for UNET3D
                if workload == 'UNET3D':
                    rdf = filter_df_by_time(rdf, "timestamp", start=fourth_epoch_start)

                bws = get_throughputs(rdf)
                lat = get_latency(rdf)
                fill(plotting_data[gpu_key][batch_key], 'vfsr_throughput', bws)
                fill(plotting_data[gpu_key][batch_key], 'vfsr_latency', lat)
            else:
                print(f'Error: missing {read_trace}')
                error = True

            write_trace = log_dir / "processed/write.out"
            if write_trace.exists():
                wdf = pd.read_csv(write_trace, on_bad_lines='skip', names=WRITE_HEADER, header=None, engine='c', delim_whitespace=True, low_memory=False, dtype={"size": float}, parse_dates=['timestamp'])
                wdf = wdf[wdf['filename'].str.contains(CKPT_FILE_PATTERN, regex=True, na=False)]
                bws = get_throughputs(wdf)
                lat = get_latency(wdf)

                fill(plotting_data[gpu_key][batch_key], 'vfsw_throughput', bws)
                fill(plotting_data[gpu_key][batch_key], 'vfsw_latency', lat)
            else:
                print(f'Error: missing {write_trace}')
                error = True

            bio_trace = log_dir / "processed/bio.out"
            if bio_trace.exists():
                bio = pd.read_csv(bio_trace, on_bad_lines='skip', names=BIO_HEADER, header=None, engine='c', delim_whitespace=True, low_memory=False, dtype={"size": float}, parse_dates=['timestamp'])        
                bior = bio[(bio['disk'] == 'sdb') & (bio['type'] == 'R')]
                biow = bio[(bio['disk'] == 'sdb') & (bio['type'] == 'W')]
                
                bws = get_throughputs(bior)
                lat = get_latency(bior)
                fill(plotting_data[gpu_key][batch_key], 'bior_throughput', bws)
                fill(plotting_data[gpu_key][batch_key], 'bior_latency', lat)

                bws = get_throughputs(biow)
                lat = get_latency(biow)
                fill(plotting_data[gpu_key][batch_key], 'biow_throughput', bws)
                fill(plotting_data[gpu_key][batch_key], 'biow_latency', lat)
            else:
                print(f'Error: missing {bio_trace}')
                error = True
    
    with open(output_dir / 'plotting_data.json', 'w') as outfile:
        json.dump(plotting_data, outfile, indent=4)

    return plotting_data


def get_throughputs(trace):
    """
    Returns the observed bandwidth in the bio trace in MB/s.
    """

    # Convert from B/ns to MB/s by multiplying by 1000:
    # B/ns * 10^9 ns/s * 10^-6 MB/B = B/ns * 10^3 ns.MB/s.B
    r = (1000*(trace['size']/trace['latency']))
    return r


def get_latency(trace):
    # convert from ns to s
    return (trace['latency'] / 1_000_000_000)

def fill(ds, metric, pd_data):
    pd_describe = pd_data.describe()
    # print(pd_describe)
    ds[metric] = {
        'mean': pd_describe['mean'],
        'std': pd_describe['std'],
        'q1': pd_describe['25%'],
        'median': pd_describe['50%'],
        'q3': pd_describe['75%'],
    }
        


def plot_throughputs(plotting_data, output_dir, workload, legend=False, title=None):
    print('Plotting Throughputs')

    metrics_to_plot_pretty_names = {
        "bior_throughput": "BIO R Throughput",
        "vfsr_throughput": "VFS R Throughput",
        "biow_throughput": "BIO W Throughput",
        "vfsw_throughput": "VFS W Throughput",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    output_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if title:
        filename = f"{workload}_{title}_throughputs.png"
    else:
        filename = f"{workload}_throughputs.png"

    figure_filename = output_dir / filename

    FONTSIZE = 16

    GPUs_to_plot = plotting_data.keys()
    plotted_batch_sizes = set()

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(5 * len(metrics_to_plot), 7)
    )

    i_ax = -1
    for metric in metrics_to_plot:
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:
            
            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
            plotted_batch_sizes.update(batches_to_plot)
            
            x = np.asarray([b for b in batches_to_plot if metric in plotting_data[gpu_key][b]])

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot if metric in plotting_data[gpu_key][batch]]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot if metric in plotting_data[gpu_key][batch]]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot if metric in plotting_data[gpu_key][batch]]
            q3 = np.asarray(q3)

            ax.plot(x, y, label=f"{gpu_key} GPU{'s' if gpu_key > 1 else ''}")

            # if metric != 'from_disk_throughput':
            ax.fill_between(x, q1, q3, alpha=0.1)


        # We support plotting different number of points for
        # different numbers of GPUs, so we keep track of the
        # longest amount of points to properly label the x axis
        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]

        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)
        # Reset for next axis
        plotted_batch_sizes = set()

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        ax.set_ylabel("MB / s", fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.25, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    plt.cla() 
    plt.close('all')   
    plt.close(fig)



def plot_latencies(plotting_data, output_dir, workload, legend=False, title=None):
    print('Plotting Throughputs')

    metrics_to_plot_pretty_names = {
        "bior_latency": "BIO R Latency",
        "vfsr_latency": "VFS R Latency",
        "biow_latency": "BIO W Latency",
        "vfsw_latency": "VFS W Latency",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    output_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if title:
        filename = f"{workload}_{title}_latencies.png"
    else:
        filename = f"{workload}_latencies.png"
    figure_filename = output_dir / filename

    FONTSIZE = 16

    GPUs_to_plot = plotting_data.keys()
    plotted_batch_sizes = set()

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(5 * len(metrics_to_plot), 7)
    )

    i_ax = -1
    for metric in metrics_to_plot:
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:
            
            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
            plotted_batch_sizes.update(batches_to_plot)
            
            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)

            ax.plot(x, y, label=f"{gpu_key} GPU{'s' if gpu_key > 1 else ''}")

            # if metric != 'from_disk_throughput':
            ax.fill_between(x, q1, q3, alpha=0.1)


        # We support plotting different number of points for
        # different numbers of GPUs, so we keep track of the
        # longest amount of points to properly label the x axis
        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]

        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)
        # Reset for next axis
        plotted_batch_sizes = set()

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        ax.set_ylabel("s", fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.25, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)



    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    plt.cla() 
    plt.close('all')   
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate average times spent in diff phases of training")
    parser.add_argument("data_dirs", nargs='+', help="Data directories")
    parser.add_argument("workload", help="Workload", choices=['unet3d', 'unet', 'dlrm'])
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory. Defaults to the data directory for single dir and 'data_step_breakdown' for multiple data directories.")
    parser.add_argument("-t", "--title", default=None, help="Additonal string to put after workload name for plots")
    parser.add_argument("-dp", "--do-processing", action="store_true", help="Whether or not to process even if plotting data is found")
    args = parser.parse_args()


    data_dirs = [Path(dir) for dir in args.data_dirs]
    workload = args.workload.upper()
    title = args.title
    do_processing = args.do_processing

    # In the usual case of a single data directory, output in it
    if len(data_dirs) == 1:
        output_dir = Path(args.data_dirs[0])
    else:
        if args.output_dir is None:
            output_dir = f"data_step_breakdown/{workload}_{int(time.time())}"
        else:
            output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)

    datafile = output_dir / 'plotting_data'

    if not do_processing and datafile.exists():
        with open(datafile, 'r') as infile:
            plotting_data = json.load(datafile)
    else:
        plotting_data = preprocess_data(data_dirs, output_dir, workload)

    plot_throughputs(plotting_data, output_dir, workload, legend=True, title=title)
    plot_latencies(plotting_data, output_dir, workload, legend=True, title=title)
