import os
import re
import json
import copy
import pathlib
import argparse
from matplotlib import patches
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data dictionary that will hold duration values for each epoch

# # For UNET3D
# metrics_pretty_names = {
#     "load_batch_mem": "1-Batch loaded in memory",
#     # "sample_load": "1.1-Sample loaded",
#     # "sample_preproc": "1.2-Sample preprocessed",
#     "load_batch_gpu": "2-Batch loaded to GPU",
#     "model_forward_pass": "3-Forward pass",
#     "loss_tensor_calc": "4-Loss calculation",
#     "model_backward_pass": "5-Backward pass",
#     "model_optim_step": "6-Optimizer step", 
#     "cum_loss_fn_calc": "7-Cumulative loss",
#     "step_end": "Overall step",
#     "all_compute": "8-Computation Only"
# }

# eval_metrics_pretty_names = {
#     "eval_step_end": "Overall eval step",
#     "eval_load_batch_mem": "1-Batch loaded in memory",
#     "eval_load_batch_gpu": "2-Batch loaded to GPU",
#     "eval_sliding_window": "3-Sliding window calc",
#     "eval_loss_and_score_fn": "4-Loss and score calc",
#     "eval_image_sizes": "Image Sizes"
# }
# durations = { metric: [] for metric in metrics_pretty_names }
# eval_durations = { metric: [] for metric in eval_metrics_pretty_names }

# WORKLOAD = "UNET"
# GPUs_int = [2, 4, 6, 8]
# # GPUs_int = [8]
# batches_int = [1, 2, 3, 4, 5]
# # batches_int = [1, 2, 4, 6, 8]
# GPUs_str = [str(g) for g in GPUs_int]
# batches_str = [ str(b) for b in batches_int ]

# # PATTERN=f'.*_ins_nostep7\.json'
# # SUFFIX=f'_ins_nostep7.json'

# PATTERN=f'.*_ins_original\.json'
# SUFFIX=f'_ins_original.json'
# NUM_EPOCHS = 50
# NUM_EVALS = 2


# For DLRM
metrics_pretty_names = {
    "step_end": "Overall step",
    "all_compute": "Computation",
    "load_batch_mem": "1-Batch loaded in memory",
    "model_forward_pass": "3-Forward pass",
    "loss_tensor_calc": "4-Loss calculation",
    "model_backward_pass": "5-Backward pass",
    "model_optim_step": "6-Optimizer step",
}
eval_metrics_pretty_names = {
    "eval_step_end": "Overall eval step",
    "eval_compute": "Eval Computation",
    "eval_load_batch_mem": "1-Batch loaded in memory",
    "eval_forward_pass": "2-Forward Pass",
    "eval_all_gather": "3-All gather",
    "eval_score_compute": "4-Score Computation",
}
durations = { metric: [] for metric in metrics_pretty_names }
eval_durations = { metric: [] for metric in eval_metrics_pretty_names }

GPUs_int = [2, 4, 6, 8]
# batches_int = [2048, 4096, 8192, 16384, 32768, 65536]
batches_int = [2048, 4096, 8192, 16384, 32768, 65536, 130712, 262144, 524288, 1048576, 2097152]
GPUs_str = [str(g) for g in GPUs_int]
# batches_str = [ str(b) for b in batches_int ]
batches_str = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M', '2M']

WORKLOAD = "DLRM"
PATTERN=f'DLRM_.*\.json'
NUM_EPOCHS = 1
NUM_EVALS = 1



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



def preprocess_data(data_dir, workload, detail=None):
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]
    log_files.sort()

    all_metrics = copy.deepcopy(durations)
    all_metrics['data_loading_bandwidth'] = []
    all_metrics['data_proc_bandwidth'] = []


    # DS for plotting data
    plotting_data = {}

    for log_file in log_files:
        print(log_file)


        gpu_key = get_num_gpus(log_file)
        batch_key = get_batch_size(log_file)
        global_batch = gpu_key * batch_key

        if gpu_key in plotting_data:
            plotting_data[gpu_key][batch_key] = {}
        else:
            plotting_data[gpu_key] = {
                batch_key: {}
            }

        all_times = copy.deepcopy(all_metrics)

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()

        # Gather all durations for each epoch in a log file
        time_to_first_forward = []
        epoch_completion = []

        for line in log:
            # Skip 1st epoch for UNET3D
            if workload != 'dlrm' and 'metadata' in line and 'epoch_num' in line['metadata'] and line['metadata']['epoch_num'] == 1:
                continue

            if line['key'] in durations.keys():
                # Append value to appropriate array
                value = round(line['value']['duration'] / 1_000_000_000, 6)

                # Define data loading bandwidth as global batch / time to load batch
                if line['key'] == 'load_batch_mem':
                    all_times['data_loading_bandwidth'].append(global_batch / value)

                if line['key'] == 'all_compute':
                    all_times['data_proc_bandwidth'].append(global_batch / value)

                if line['key'] == 'step_end':
                    # sum_others = round(sum_others, 3)
                    all_times['step_end'].append(value)

                else:
                    all_times[line['key']].append(value)

        
        for metric in all_times:
            median = stats.median(all_times[metric])
            quartiles = stats.quantiles(all_times[metric])

            plotting_data[gpu_key][batch_key][metric] = {
                'median': median,
                'q1': quartiles[0],
                'q3': quartiles[2],
            }


    print("Plotting step breakdown from raw data for epochs > 1")

    ###############################################################
    # Modify metrics, gpus or batch sizes to plot here
    ###############################################################

    metrics_to_plot_pretty_names = {
        "data_loading_bandwidth": "Data Loading",
        "data_proc_bandwidth": "Data Processing",
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
            ax.set_ylabel("BW (samples/s)", fontsize=FONTSIZE)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.35, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    # fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_bandwidths.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


    metrics_to_plot_pretty_names = {
        "step_end": "Overall",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calc",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
        # "cum_loss_fn_calc": "2.5 Cumulative loss",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5))

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        if i_ax == 4:
            ax.spines['right'].set_visible(False)

        if i_ax >= 5:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left = False)
            # ax.yaxis.tick_params(which="major", visible=False)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:

            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
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

            ax.set_xscale('log', base=2)
            # for large batch sizes
            if workload == 'dlrm':
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            else:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, ha='center', fontsize=FONTSIZE-2)

            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            ax.set_xlabel("Batch size", fontsize=FONTSIZE)
            ax.set_ylabel("Time (s)", fontsize=FONTSIZE)
            # ax.legend(fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.01, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    # fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_breakdown_raw.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


    print(f'Printing smaller plot')
    ## Shorter plot 

    metrics_pretty_names = {
        "step_end": "Overall",
        "load_batch_mem": "Loading",
        "all_compute": "Computation",
        # "sum_computation": "Computation",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 6), sharey=True)

    FONTSIZE = 18
    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        if i_ax == 0:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if i_ax >= 1:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left = False)
            # ax.yaxis.tick_params(which="major", visible=False)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:
            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
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
            # ax.set_xscale('log', base=2)

            if len(batches_to_plot_str[0]) > 3:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-46, ha='center', fontsize=FONTSIZE)
                # ax.tick_params(axis='x', labelrotation=45)
            else:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, ha='center', fontsize=FONTSIZE-1)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.2, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch size', fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_breakdown_paper.png"
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
    parser.add_argument("workload", help="Workload", choices=['unet3d', 'dlrm'])
    parser.add_argument("-n", "--name", help="Detailed name to differentiate plots")
    args = parser.parse_args()


    # export_per_epoch_stats(args.data_dir)
    # export_per_eval_stats(args.data_dir)

    preprocess_data(args.data_dir, args.workload, args.name)

    # if args.workload == 'unet3d':
    #     UNET_export_overall_epoch_stats(args.data_dir)
    # else:
    #     DLRM_export_overall_epoch_stats(args.data_dir)
    
    # export_overall_eval_stats(args.data_dir)

    # plot_latency_histograms(args.data_dir)

    # if args.workload == 'unet3d':
    # UNET_plot_epoch_step_breakdown_paper(args.data_dir, args.name)
    #     UNET_plot_epoch_step_breakdown_paper(args.data_dir, args.name)
    # else:
    #     DLRM_plot_epoch_individual_time_curves(args.data_dir, sharey=True)
    
    # plot_eval_individual_time_curves(args.data_dir)

    # UNET_plot_epoch_individual_time_curves(expdir, sharey=True, median=False)
    # plot_eval_individual_time_curves(expdir, sharey=True, median=False)


    # plot_eval_individual_time_curves_by_image_size(expdir, sharey=True)
