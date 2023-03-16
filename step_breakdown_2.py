import os
import re
import json
import copy
import pathlib
import argparse
import time
from matplotlib import patches
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pprint import pprint

# Data dictionary that will hold duration values for each epoch


METRICS_PER_WORKLOAD = {
    "UNET3D": {
        "step_end": "Overall step",
        "load_batch_mem": "1 Batch load to mem",
        "all_compute": "2 Computation",
        "load_batch_gpu": "2.1 Batch load to GPU",
        "model_forward_pass": "2.2 Forward pass",
        "loss_tensor_calc": "2.3 Loss calculation",
        "model_backward_pass": "2.4 Backward pass",
        "model_optim_step": "2.5 Optimizer step", 
        "cum_loss_fn_calc": "2.6 Cumulative loss",
    },
    "DLRM": {
        "step_end": "Overall step",
        "load_batch_mem": "1 Batch load to mem",
        "all_compute": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calculation",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
    }
}

# Global variable to create a new plotting directory
timestamp = int(time.time())
plotting_dir = f'plots_{timestamp}'


DLRM_BATCH_SIZE_STRINGS = {
    2048: "2k",
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
    65536: "64k",
    130712: "128k",
    262144: "256k",
    524288: "512k",
    1048576: "1M",
    2097152: "2M",
}



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
    return int(res.group(1))

def get_num_workers(log_file_name):
    res = re.search(r'.*([0-9]+)workers.*', log_file_name)
    if res is None:
        res = re.search(r'.*([0-9]+)w_.*', log_file_name)
    if res is None:
        res = re.search(r'.*w([0-9]+).*', log_file_name)
    return int(res.group(1))



def preprocess_data(data_dir, workload):
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]
    log_files.sort()

    metrics_pretty_names = METRICS_PER_WORKLOAD[workload]
    events_of_interest = set(metrics_pretty_names)

    durations = { metric: [] for metric in metrics_pretty_names }
    all_metrics = copy.deepcopy(durations)

    all_metrics['data_loading_throughput'] = []
    all_metrics['data_proc_throughput'] = []

    outfile = open(os.path.join(data_dir, f"{workload}_step_analysis.txt"), "w")
    outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Std':>15}\t{'Median':>15}\t{'q1':>15}\t{'q3':>15}\n")


    # DS for plotting data
    plotting_data = {}

    for log_file in log_files:
        print(log_file)
        outfile.write(f"{log_file}\n")

        gpu_key = get_num_gpus(log_file)
        batch_key = get_batch_size(log_file)

        if gpu_key in plotting_data:
            plotting_data[gpu_key][batch_key] = {}
        else:
            plotting_data[gpu_key] = {
                batch_key: {}
            }

        current_file_times = copy.deepcopy(all_metrics)

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()

        # Gather all durations for each epoch in a log file
        time_to_first_forward = []
        epoch_completion = []

        epoch = 0
        seen_events = set()

        for line in log:

            event = line['key']

            # Skip the first step every epoch
            if event == 'epoch_start':
                epoch = line['metadata']['epoch_num']
                seen_events = set()

            # Skip first epoch for unet3d
            if workload == 'UNET3D' and epoch == 1:
                continue

            # For DLRM, we'll skip the first step when a new training block starts
            if event == 'training_start':
                seen_events = set()


            if event in events_of_interest:

                # Skip the first step, i.e. wait until we've seen all events once
                if seen_events != events_of_interest:
                    seen_events.add(event)
                    continue

                # Append value to appropriate array
                value = round(line['value']['duration'] / 1_000_000_000, 6)

                # Define data loading bandwidth as global batch / time to load batch
                if event == 'load_batch_mem':
                    current_file_times['data_loading_throughput'].append(batch_key / value)

                if event == 'all_compute':
                    current_file_times['data_proc_throughput'].append(batch_key / value)

                if event == 'step_end':
                    current_file_times['step_end'].append(value)

                else:
                    current_file_times[event].append(value)

        
        for key in current_file_times:
            mean = stats.mean(current_file_times[key])
            median = stats.median(current_file_times[key])
            std = stats.stdev(current_file_times[key])
            quartiles = stats.quantiles(current_file_times[key])

            plotting_data[gpu_key][batch_key][key] = {
                'mean': mean,
                'std': std,
                'median': median,
                'q1': quartiles[0],
                'q3': quartiles[2],
            }
            ROUND = 5
            outfile.write(f"{key:>30}:\t{round(mean, ROUND):>15}\t{round(median, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
        outfile.write("\n")



    print('Exporting simulation sleep times')
    simulation_sleep_time = {}

    for gpu_key in plotting_data:
        for batch_key in plotting_data[gpu_key]:
            if gpu_key in simulation_sleep_time:
                simulation_sleep_time[gpu_key][batch_key] = {
                    'mean': plotting_data[gpu_key][batch_key]['all_compute']['mean'],
                    'std': plotting_data[gpu_key][batch_key]['all_compute']['std'],
                    'median': plotting_data[gpu_key][batch_key]['all_compute']['median'],
                    'q1': plotting_data[gpu_key][batch_key]['all_compute']['q1'],
                    'q3': plotting_data[gpu_key][batch_key]['all_compute']['q3'],
                }
            else:
                simulation_sleep_time[gpu_key] = {
                    batch_key: {
                        'mean': plotting_data[gpu_key][batch_key]['all_compute']['mean'],
                        'std': plotting_data[gpu_key][batch_key]['all_compute']['std'],
                        'median': plotting_data[gpu_key][batch_key]['all_compute']['median'],
                        'q1': plotting_data[gpu_key][batch_key]['all_compute']['q1'],
                        'q3': plotting_data[gpu_key][batch_key]['all_compute']['q3'],
                    }
                }     

    with open(os.path.join(data_dir, f"{workload}_sleep_times_{timestamp}.json"), "w") as outfile:
        json.dump(simulation_sleep_time, outfile, indent=4)
    
    return plotting_data



def plot_throughputs(plotting_data, workload):
    print('Plotting Throughputs')

    metrics_to_plot_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Processing Throughput",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

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

            ax.plot(x, y, label=f"{gpu_key} GPUs")
            ax.fill_between(x, q1, q3, alpha=0.1)


        # We support plotting different number of points for
        # different numbers of GPUs, so we keep track of the
        # longest amount of points to properly label the x axis
        plotted_batch_sizes = sorted(list(plotted_batch_sizes))

        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        ax.set_ylabel("Samples / s", fontsize=FONTSIZE)
        # Reset for next axis
        plotted_batch_sizes = set()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.25, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    output_dir = os.path.join(data_dir, plotting_dir)
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{workload}_throughputs.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    plt.cla() 
    plt.close('all')   
    plt.close(fig)



def plot_step_breakdown(plotting_data, workload, sharey=True):
    print(f'Plotting Step breakdown summary')

    ## Shorter plot 
    metrics_pretty_names = {
        "step_end": "Overall Step",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot), 6), 
        sharey=sharey
    )

    GPUs_to_plot = plotting_data.keys()
    plotted_batch_sizes = set()

    FONTSIZE = 18
    i_ax = -1

    for metric in metrics_to_plot:
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
            plotted_batch_sizes.update(batches_to_plot)

            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            ax.fill_between(x, q1, q3, alpha=0.1)

        plotted_batch_sizes = sorted(list(plotted_batch_sizes))

        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-3)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)
        plotted_batch_sizes = set()


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='upper right',
        bbox_to_anchor = (-0.065, -0.05, 1, 1), 
        bbox_transform = plt.gcf().transFigure, 
        fontsize=FONTSIZE
    )

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, plotting_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{workload}_step_breakdown{'' if sharey else '_indiv'}.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)




def plot_full_breakdown(plotting_data, workload, sharey=True):
    print(f'Plotting full step breakdown (sharey={sharey})')

    metrics_to_plot_pretty_names = {
        "step_end": "Overall",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calc",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
    }
    if workload == 'UNET3D':
        metrics_to_plot_pretty_names["cum_loss_fn_calc"] = "2.5 Cumulative loss"

    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = plotting_data.keys()
    plotted_batch_sizes = set()

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5), sharey=sharey)

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        if sharey:
            if i_ax == 0:
                ax.set_ylabel("Time (s)", fontsize=FONTSIZE)

            if i_ax == 2:
                ax.spines['right'].set_visible(False)

            if i_ax >= 3:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left = False)

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
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.1)
        
        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-3)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        plotted_batch_sizes = set()

    handles, labels = ax.get_legend_handles_labels()

    if sharey:
        fig.legend(
            handles, 
            labels, 
            loc='upper right',  
            bbox_to_anchor = (-0.01, -0.05, 1, 1), 
            bbox_transform = plt.gcf().transFigure, 
            fontsize=FONTSIZE-1
        )
    else:
        fig.legend(
            handles, 
            labels, 
            fontsize=FONTSIZE-1
        )

    output_dir = os.path.join(data_dir, plotting_dir)
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{workload}_full_breakdown{'' if sharey else '_indiv'}.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)




def plot_full_breakdown_and_throughputs(plotting_data, workload, sharey=True):
    print(f'Plotting throughputs and full step breakdown (sharey={sharey})')

    metrics_to_plot_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Processing Throughput",
        "step_end": "Overall",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calc",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
    }
    if workload == 'UNET3D':
        metrics_to_plot_pretty_names["cum_loss_fn_calc"] = "2.5 Cumulative loss"

    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = plotting_data.keys()
    plotted_batch_sizes = set()

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5))

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        if sharey:

            if i_ax == 0:
                ax.set_ylabel("Samples / s", fontsize=FONTSIZE)

            if i_ax == 2:
                ax.spines['right'].set_visible(False)
                ax.set_ylabel("Time (s)", fontsize=FONTSIZE)

            if i_ax >= 3:
                ax.sharey(axes[2])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left = False)

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
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.1)

        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-10)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(batches_to_plot, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        plotted_batch_sizes = set()


    handles, labels = ax.get_legend_handles_labels()
    
    # if sharey:
    #     fig.legend(
    #         handles, 
    #         labels, 
    #         loc='upper right', 
    #         bbox_to_anchor = (-0.01, -0.05, 1, 1), 
    #         bbox_transform = plt.gcf().transFigure, 
    #         fontsize=FONTSIZE-1
    #     )
    # else:
    #     # fig.legend(
    #     #     handles, 
    #     #     labels, 
    #     #     fontsize=FONTSIZE-1
    #     # )
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, plotting_dir)
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{workload}_full_breakdown_throughputs{'' if sharey else '_indiv'}.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate average times spent in diff phases of training")
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("workload", help="Workload", choices=['unet3d', 'dlrm'])
    args = parser.parse_args()

    data_dir = args.data_dir
    workload = args.workload.upper()

    plotting_data = preprocess_data(data_dir, workload)

    plot_throughputs(plotting_data, workload)

    plot_step_breakdown(plotting_data, workload)
    plot_step_breakdown(plotting_data, workload, sharey=False)

    plot_full_breakdown(plotting_data, workload)
    plot_full_breakdown(plotting_data, workload, sharey=False)

    plot_full_breakdown_and_throughputs(plotting_data, workload)
    plot_full_breakdown_and_throughputs(plotting_data, workload, sharey=False)

