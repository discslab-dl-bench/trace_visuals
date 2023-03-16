import os
import re
import json
import pathlib
import argparse
from matplotlib import patches
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path



PATTERN_INSTRU_LINE = re.compile(r'\[INFO\] (load_batch_mem|all_compute|step_end|) (\d+)')



def get_smallest_common_length(data):
    l = float("inf")
    for k,v in data.items():
        if len(v) != l:
            l = min(l, len(v))
    return l


def get_num_gpus(log_file_name):
    res = re.search(r'.*([0-9])GPU.*', log_file_name)
    if res is None:
        res = re.search(r'.*([0-9])gpu.*', log_file_name)
    if res is None:
        res = re.search(r'.*([0-9])g.*', log_file_name)
    return int(res.group(1))


def get_batch_size(log_file_name):
    res = re.search(r'.*([0-9]+)batch.*', log_file_name)
    if res is None:
        res = re.search(r'.*_([0-9]+)b_.*', log_file_name)
    if res is None:
        res = re.search(r'.*batch([0-9]+).*', log_file_name)
    if res is None:
        res = re.search(r'.*b([0-9]+).*', log_file_name)
    return int(res.group(1))

def get_num_workers(log_file_name):
    res = re.search(r'.*([0-9]+)workers.*', log_file_name)
    if res is None:
        res = re.search(r'.*([0-9]+)w_.*', log_file_name)
    if res is None:
        res = re.search(r'.*w([0-9]+).*', log_file_name)
    return int(res.group(1))


PATTERN_START_EPOCH = re.compile(r'.*Starting epoch ([0-9]+)')




def generate_plotting_data(data_dir, workload):

    data_dir = Path(data_dir)

    log_files = sorted(list(data_dir.rglob('dlio.log')))

    outfile = open(os.path.join(data_dir, f"dlio_{workload}_step_analysis.txt"), "w")
    outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}\n")

    plotting_data = {}

    for log_file in log_files:
        outfile.write(f"{log_file}\n")

        experiment_name = log_file.parts[1]
        print(experiment_name)

        num_gpu = get_num_gpus(experiment_name)
        batch_size = get_batch_size(experiment_name)
        # global_batch_size = num_gpu * batch_size
        global_batch_size = batch_size

        if num_gpu not in plotting_data:
            plotting_data[num_gpu] = {}
        
        epoch = 0
        seen_evts = set()
        with open(log_file, 'r') as log_file:

            current_file_data = {
                "data_loading_throughput": [],
                "data_proc_throughput": [],
                "load_batch_mem": [],
                "all_compute": [],
                "step_end": [],
            }

            # TODO: Don't consider first step 

            for line in log_file:

                if m := re.match(PATTERN_START_EPOCH, line):
                    print(line)
                    epoch = int(m.group(1))
                    seen_evts = set()

                if workload == 'unet3d':
                    if epoch == 1:
                        continue
                
                if m := re.match(PATTERN_INSTRU_LINE, line):
                    event = m.group(1)
                    duration = int(m.group(2)) / 1_000_000_000   # convert to seconds

                    # Skip the first step by making sure we've seen the 3 events
                    # everytime we're in a new epoch
                    if len(seen_evts) < 3:
                        print(f'Skipping line {event} {duration}')
                        seen_evts.add(event)
                        continue

                    current_file_data[event].append(duration)

                    if event == 'load_batch_mem':
                        data_tp = global_batch_size / duration
                        current_file_data['data_loading_throughput'].append(data_tp)
                    elif event == 'all_compute':
                        proc_tp = global_batch_size / duration
                        current_file_data['data_proc_throughput'].append(proc_tp)
        

        plotting_data[num_gpu][batch_size] = {}
        
        for key in current_file_data:
            print(key)
            mean = stats.mean(current_file_data[key])
            median = stats.median(current_file_data[key])
            std = stats.stdev(current_file_data[key])
            quartiles = stats.quantiles(current_file_data[key])

            plotting_data[num_gpu][batch_size][key] = {
                'median': median,
                'q1': quartiles[0],
                'q3': quartiles[2],
            }
            ROUND = 5
            outfile.write(f"{key:>30}:\t{round(mean, ROUND):>15}\t{round(median, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
        outfile.write("\n")

    outfile.flush()
    outfile.close()

    from  pprint import pprint
    pprint(plotting_data)

    return plotting_data



def plot_throughputs(data_dir, output_dir, workload):

    GPUs_int = []
    batches_int = []

    for num_gpu in plotting_data:
        GPUs_int.append(int(num_gpu))

    for batch_size in plotting_data[GPUs_int[0]]:
        batches_int.append(int(batch_size)) 

    GPUs_int.sort()
    batches_int.sort()
    GPUs_str = [str(g) for g in GPUs_int]
    batches_str = [str(b) for b in batches_int]

    print("Plotting step breakdown from raw data for epochs > 1")

    ###############################################################
    # Modify metrics, gpus or batch sizes to plot here
    ###############################################################

    metrics_to_plot_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Processing Throughput",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str

    FONTSIZE = 16

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(5 * len(metrics_to_plot.keys()), 7)
    )

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:

            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
            if workload == 'dlrm':
                batches_to_plot_str = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k']
            else:
                batches_to_plot_str = [str(b) for b in batches_to_plot]

            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)

            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.1)

            # for large batch sizes
            if workload == 'dlrm':
                ax.set_xscale('log', base=2)
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            else:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, ha='center', fontsize=FONTSIZE-2)

            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            ax.set_xlabel("Batch size", fontsize=FONTSIZE)
            ax.set_ylabel("Samples / s", fontsize=FONTSIZE)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.2, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    # fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"dlio_{workload}_throughputs.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



def plot_full_step_breakdown(data_dir, output_dir, workload):
    metrics_to_plot_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Processing Throughput",
        "step_end": "Overall Step",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_int = []
    batches_int = []

    for num_gpu in plotting_data:
        GPUs_int.append(int(num_gpu))

    for batch_size in plotting_data[GPUs_int[0]]:
        batches_int.append(int(batch_size)) 

    GPUs_int.sort()
    batches_int.sort()
    GPUs_str = [str(g) for g in GPUs_int]
    batches_str = [str(b) for b in batches_int]

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot.keys()), 5)
    )

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)
        
        if i_ax == 0:
            ax.set_ylabel("Samples / s", fontsize=FONTSIZE)


        if i_ax == 2:
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("Time (s)", fontsize=FONTSIZE)

        if i_ax >= 3:
            ax.sharey(axes[2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left = False)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:

            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
            if workload == 'dlrm':
                batches_to_plot_str = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k']
            else:
                batches_to_plot_str = [str(b) for b in batches_to_plot]

            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.1)

            # for large batch sizes
            if workload == 'dlrm':
                ax.set_xscale('log', base=2)
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            else:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, ha='center', fontsize=FONTSIZE-2)

            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            ax.set_xlabel("Batch size", fontsize=FONTSIZE)
            # ax.legend(fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.01, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    # fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"dlio_{workload}_full_step_breakdown.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_step_breakdown(data_dir, output_dir, workload, sharey=True):

    print(f'Printing smaller plot')
    ## Shorter plot 

    metrics_pretty_names = {
        "step_end": "Overall Step",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }

    GPUs_int = []
    batches_int = []

    for num_gpu in plotting_data:
        GPUs_int.append(int(num_gpu))

    for batch_size in plotting_data[GPUs_int[0]]:
        batches_int.append(int(batch_size)) 

    GPUs_int.sort()
    batches_int.sort()
    GPUs_str = [str(g) for g in GPUs_int]
    batches_str = [str(b) for b in batches_int]

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str


    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot.keys()), 6), 
        sharey=sharey)

    FONTSIZE = 18
    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        if sharey:
            if i_ax == 0:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if i_ax >= 1:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left = False)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:
            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))

            if workload == 'dlrm':
                batches_to_plot_str = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k']
            else:
                batches_to_plot_str = [str(b) for b in batches_to_plot]

            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            ax.fill_between(x, q1, q3, alpha=0.1)

            if workload == 'dlrm':
                ax.set_xscale('log', base=2)
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-46, ha='center', fontsize=FONTSIZE)
            else:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, ha='center', fontsize=FONTSIZE-1)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)


    handles, labels = ax.get_legend_handles_labels()
    
    legend_x_offset = -0.075 if sharey else -0.06

    fig.legend(handles, labels, 
        loc='upper right',  
        bbox_to_anchor = (legend_x_offset, -0.05, 1, 1), 
        bbox_transform = plt.gcf().transFigure, 
        fontsize=FONTSIZE
    )

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch size', fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"dlio_{workload}_step_breakdown{'_sharey' if sharey else ''}.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)





if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate average times spent in diff phases of training")
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("workload", help="Workload", choices=['unet3d', 'bert', 'dlrm'])
    parser.add_argument("-o", "--output-dir", help="Output directory", default='data_step_processed')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    workload = args.workload

    plotting_data = generate_plotting_data(data_dir, workload)

    plot_throughputs(data_dir, output_dir, workload)
    plot_step_breakdown(data_dir, output_dir, workload)
    plot_step_breakdown(data_dir, output_dir, workload, sharey=False)

