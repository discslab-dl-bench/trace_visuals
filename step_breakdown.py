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
batches_int = [2048, 4096, 8192, 16384, 32768, 65536, 130712, 262144]
# batches_int = [2048, 4096, 8192, 16384, 32768, 65536, 130712, 262144, 524288, 1048576, 2097152]
GPUs_str = [str(g) for g in GPUs_int]
# batches_str = [ str(b) for b in batches_int ]
batches_str = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k']
# batches_str = ['2k', '4k', '8k', '16k', '32k', '64k', '128k', '256k', '512k', '1M', '2M']

WORKLOAD = "DLRM"
PATTERN=f'DLRM_.*\.json'
NUM_EPOCHS = 1
NUM_EVALS = 1

SKIP_N_LINES = 100

def export_per_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]
    log_files.sort()

    for log_file in log_files:
        print(f"Processing {log_file}")

        per_epoch_durations = {
            n + 1: copy.deepcopy(durations) for n in range(0, NUM_EPOCHS)
        }
        per_epoch_stats = {
            n + 1: copy.deepcopy(durations) for n in range(0, NUM_EPOCHS)
        }

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()



        # Gather all durations for each epoch in a log file
        for i, line in enumerate(log):

            if i < SKIP_N_LINES:
                continue

            if line['key'] == "epoch_start":
                epoch = line['metadata']['epoch_num']
                continue

            # keys: time_ms, event_type, key, value, metadata
            if line['key'] in durations.keys():
                # Append value to appropriate array
                value = line['value']['duration']
                per_epoch_durations[epoch][line['key']].append(value)

        # Calculate stats for each epoch in current log file
        for epoch, data in per_epoch_durations.items():
            for metric in durations.keys():
                per_epoch_stats[epoch][metric] = {
                    "mean": int(stats.mean(data[metric])),
                    "stdev": int(stats.pstdev(data[metric])),
                }
        # Gets a prettier name from the log file name
        outfile_base = os.path.basename(log_file).replace(".json", "")

        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, "per_epoch")).mkdir(parents=True, exist_ok=True)

        outfile_stats = outfile_base + "_per_epoch_stats.json"
        outfile_durations = outfile_base + "_per_epoch_durations.json"

        json.dump(per_epoch_stats, open(os.path.join(data_dir, "per_epoch", outfile_stats), mode="w"), indent=4)
        json.dump(per_epoch_durations, open(os.path.join(data_dir, "per_epoch", outfile_durations), mode="w"), indent=4)
        


def sanity_check(data_dir, out_dir):
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data')) if re.match(PATTERN, f)]
    log_files.sort()

    print(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}")

    for log_file in log_files:
        print(f"{log_file}")

        # all_times = copy.deepcopy(durations)

        all_times = {
            "step_end": [],
            "sum_others": [],
            "all_compute": [],
            "diffs": [],
        }

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()

        # Gather all durations for each epoch in a log file
        sum_others = 0
        all_compute = 0

        for line in log:
            if line['key'] == "epoch_start":
                epoch = line['metadata']['epoch_num']
                if epoch == 1 or epoch == "1":
                    continue

            if line['key'] in durations.keys():
                # Append value to appropriate array
                value = round(line['value']['duration'] / 1_000_000_000, 3)

                if line['key'] == 'sample_load' or line['key'] =='sample_preproc':
                    continue

                if line['key'] == 'step_end':
                    sum_others = round(sum_others, 3)
                    all_times['step_end'].append(value)
                    all_times['sum_others'].append(sum_others)
                    all_times['all_compute'].append(all_compute)
                    # print(f"step: {value}\tsum: {sum_others}\tdiff: {round(value - sum_others, 3)}") 
                    # Reset values
                    sum_others = 0
                    all_compute = 0
                else:
                    # print(line['key'])
                    sum_others += value
                    if line['key'] != 'load_batch_mem' and line['key'] != 'load_batch_gpu':
                        all_compute += value
            
        all_times['diffs'] = np.asarray(all_times['step_end']) - np.asarray(all_times['sum_others'])
        all_times['diffs'] = all_times['diffs'].tolist()
        
        import statistics

        for key in all_times:
            avg = round(statistics.mean(all_times[key]), 4)
            median = round(statistics.median(all_times[key]), 4)
            std = round(statistics.stdev(all_times[key]), 4)
            quantiles = statistics.quantiles(all_times[key])

            print(f"{key:>30}:\t{avg:>15}\t{median:>15}\t{std:>15}\t{round(quantiles[0], 4):>15}\t{round(quantiles[2], 4):>15}")

        outfile_base = os.path.basename(log_file).replace(SUFFIX, "").replace(".json", "")
        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, out_dir, "sanity_check")).mkdir(parents=True, exist_ok=True)
        outfile_all_times = outfile_base + "all_times.json"
        json.dump(all_times, open(os.path.join(data_dir, out_dir, "sanity_check", outfile_all_times), mode="w"), indent=4)





def plot_from_raw(data_dir, workload, detail=None):
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]
    log_files.sort()

    all_metrics = copy.deepcopy(durations)
    all_metrics["sum_computation"] = []
    all_metrics["step_summed"] = []
    all_metrics["sum_loading"] = []
    all_metrics['sum_all_compute_and_load_gpu'] = []
    all_metrics['diff_step_vs_summed'] = []
    all_metrics['data_loading_bandwidth'] = []
    # all_metrics['time_to_first_forward'] = []


    simulation_sleep_time = {}

    # DS for plotting data
    plotting_data = {}
    for num_GPU in GPUs_int:
        plotting_data[num_GPU] = {
            batch_num: copy.deepcopy(all_metrics) for batch_num in batches_int
        }
    
    with open(os.path.join(data_dir, "step_analysis.txt"), "w") as outfile:

        outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}\n")

        for log_file in log_files:
            print(log_file)
            outfile.write(f"{log_file}\n")
            gpu_key = get_num_gpus(log_file)
            batch_key = get_batch_size(log_file)

            if gpu_key not in GPUs_int or batch_key not in batches_int:
                continue

            global_batch = gpu_key * batch_key

            print(gpu_key, batch_key)
            # batch_key = get_num_workers(log_file)

            all_times = copy.deepcopy(all_metrics)

            infile = open(log_file, mode='r')
            log = json.load(infile)
            infile.close()

            # Gather all durations for each epoch in a log file
            step_summed = sum_computation = sample_load = sample_preproc = 0
            sum_all_compute_and_load_gpu = sum_loading = 0

            time_to_first_forward = []
            epoch_completion = []

            for i, line in enumerate(log):

                if i < SKIP_N_LINES:
                    continue

                # Skip 1st epoch for UNET3D
                if workload != 'dlrm' and 'metadata' in line and 'epoch_num' in line['metadata'] and line['metadata']['epoch_num'] == 1:
                    continue

                if line['key'] in durations.keys():
                    # Append value to appropriate array
                    value = round(line['value']['duration'] / 1_000_000_000, 6)

                    if line['key'] == 'sample_load':
                        sample_load += value
                        continue
                    if line['key'] =='sample_preproc':
                        sample_preproc += value
                        continue

                    if line['key'] == 'load_batch_mem':
                        all_times['data_loading_bandwidth'].append(global_batch / value)

                    if line['key'] == 'step_end':
                        # sum_others = round(sum_others, 3)
                        all_times['step_end'].append(value)
                        all_times['sum_computation'].append(sum_computation)
                        all_times['sum_all_compute_and_load_gpu'].append(sum_all_compute_and_load_gpu)

                        if "sample_load" in all_metrics and "sample_preproc" in all_metrics:
                            all_times['sample_load'].append(sample_load)
                            all_times['sample_preproc'].append(sample_preproc)

                        all_times['step_summed'].append(step_summed)
                        all_times['sum_loading'].append(sum_loading)

                        # print(f"step: {value}\tsum: {sum_others}\tdiff: {round(value - sum_others, 3)}") 
                        # Reset values
                        step_summed = sum_computation = sample_load = sample_preproc = sum_loading = 0
                        sum_all_compute_and_load_gpu = 0
                    else:
                        all_times[line['key']].append(value)

                        if line['key'] != 'all_compute':
                            step_summed += value

                            if line['key'] != 'load_batch_mem':
                                sum_computation += value

                        if line['key'] == 'load_batch_mem' or line['key'] == 'load_batch_gpu':
                            sum_loading += value

                        if line['key'] == 'load_batch_gpu' or line['key'] == 'all_compute':
                            sum_all_compute_and_load_gpu += value


            all_times['diff_step_vs_summed'] = np.asarray(all_times['step_end']) - np.asarray(all_times['step_summed'])
            all_times['diff_step_vs_summed'] = all_times['diff_step_vs_summed'].tolist()
            
            for key in all_times:
                mean = stats.mean(all_times[key])
                median = stats.median(all_times[key])
                std = stats.stdev(all_times[key])
                quartiles = stats.quantiles(all_times[key])
                
                plotting_data[gpu_key][batch_key][key] = {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'q1': quartiles[0],
                    'q3': quartiles[2],
                }
                ROUND = 4
                outfile.write(f"{key:>30}:\t{round(mean, ROUND):>15}\t{round(median, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
            outfile.write("\n")

    print("Plotting step breakdown from raw data for epochs > 1")

    simulation_sleep_time = {}
    for gpu_key in GPUs_int:
        for batch_key in batches_int:
            print(gpu_key, batch_key)
            if gpu_key in simulation_sleep_time:

                print(plotting_data[gpu_key][batch_key])
                print(plotting_data[gpu_key][batch_key]['all_compute'])
                simulation_sleep_time[gpu_key][batch_key] = {
                    'mean': plotting_data[gpu_key][batch_key]['all_compute']['mean'],
                    'std': plotting_data[gpu_key][batch_key]['all_compute']['std'],
                }
            else:
                simulation_sleep_time[gpu_key] = {
                    batch_key: {
                        'mean': plotting_data[gpu_key][batch_key]['all_compute']['mean'],
                        'std': plotting_data[gpu_key][batch_key]['all_compute']['std'],
                    }
                }     

    with open(os.path.join(data_dir, "dlio_sleep_times.json"), "w") as outfile:
        json.dump(simulation_sleep_time, outfile, indent=4)

    
    exit()

    ###############################################################
    # Modify metrics, gpus or batch sizes to plot here
    ###############################################################

    metrics_to_plot_pretty_names = {
        # "epoch_completion": "Time per epoch (>1)",
        "data_loading_bw": "Data Loading BW (samples/s)",
        "data_processing_bw": "Data Processing BW (samples/s)",
        # "step_end": "Overall",
        # "load_batch_mem": "1 Batch Loading",
        # "all_compute": "2 Computation",
        # "model_forward_pass": "2.1 Forward pass",
        # "loss_tensor_calc": "2.2 Loss calc",
        # "model_backward_pass": "2.3 Backward pass",
        # "model_optim_step": "2.4 Optimizer step",
        # "cum_loss_fn_calc": "2.5 Cumulative loss",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str

    FONTSIZE = 16

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(5 * len(metrics_to_plot.keys()), 7), sharey=False)
    # fig.suptitle(f"{WORKLOAD}{' ' + detail + ' ' if detail else ' '}Step Breakdown (epoch > 1)", fontsize=FONTSIZE)

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

            if metric == 'data_loading_bw':
                x = np.asarray(batches_to_plot)
                y = []
                q1 = []
                q3 = []

                for batch_size in batches_to_plot:
                    print(gpu_key, batch_size)
                    # Not all GPUs have all batches - can crash
                    if len(plotting_data[gpu_key][batch_size]['load_batch_mem']) > 0:
                        median_loading_time = plotting_data[gpu_key][batch_size]['load_batch_mem']["median"]
                        q1_loading_time = plotting_data[gpu_key][batch_size]['load_batch_mem']["q1"]
                        q3_loading_time = plotting_data[gpu_key][batch_size]['load_batch_mem']["q3"]
                        global_batch = batch_size * gpu_key
                        print(f'{gpu_key}: loaded {global_batch} globally in median {median_loading_time} s')

                        y.append(global_batch / median_loading_time)
                        q1.append(global_batch / q1_loading_time)
                        q3.append(global_batch / q3_loading_time)
                
                y = np.asarray(y)
                q1 = np.asarray(q1)
                q3 = np.asarray(q3)
                print(y.shape)
                print(q1.shape)
                print(q3.shape)

                if x.shape[0] != y.shape[0]:
                    diff = x.shape[0] - y.shape[0] 
                    x = x[:-diff]

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

            elif metric == 'data_processing_bw':
                x = np.asarray(batches_to_plot)
                y = []
                q1 = []
                q3 = []

                for batch_size in batches_to_plot:
                    # Not all GPUs have all batches - can crash
                    if len(plotting_data[gpu_key][batch_size]['load_batch_mem']) > 0:
                        median_loading_time = plotting_data[gpu_key][batch_size]['all_compute']["median"]
                        q1_loading_time = plotting_data[gpu_key][batch_size]['all_compute']["q1"]
                        q3_loading_time = plotting_data[gpu_key][batch_size]['all_compute']["q3"]
                        print(f'{gpu_key}: processed {batch_size} in median {median_loading_time} s')

                        global_batch = batch_size * gpu_key
                        y.append(global_batch / median_loading_time)
                        q1.append(global_batch / q1_loading_time)
                        q3.append(global_batch / q3_loading_time)
                
                y = np.asarray(y)
                q1 = np.asarray(q1)
                q3 = np.asarray(q3)


                if x.shape[0] != y.shape[0]:
                    diff = x.shape[0] - y.shape[0] 
                    x = x[:-diff]

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
        # "epoch_completion": "Time per epoch (>1)",
        # "data_loading_bw": "Data Loading BW (samples/s)",
        # "data_processing_bw": "Data Processing BW (samples/s)",
        "step_end": "Overall",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calc",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
        "cum_loss_fn_calc": "2.5 Cumulative loss",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5), sharey=True)
    # fig.suptitle(f"{WORKLOAD}{' ' + detail + ' ' if detail else ' '}Step Breakdown (epoch > 1)", fontsize=FONTSIZE)

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

            x = np.asarray(batches_to_plot)

            y = q1 = q3 = []

            # y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            # y = np.asarray(y)

            # q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            # q1 = np.asarray(q1)

            # q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            # q3 = np.asarray(q3)

            for batch_size in batches_to_plot:
                # Not all GPUs have all batches - can crash
                if len(plotting_data[gpu_key][batch_size][metric]) > 0:
                    median_val = plotting_data[gpu_key][batch_size][metric]["median"]
                    q1_val = plotting_data[gpu_key][batch_size][metric]["q1"]
                    q3_val = plotting_data[gpu_key][batch_size][metric]["q3"]

                    y.append(median_val)
                    q1.append(q1_val)
                    q3.append(q3_val)
            
            y = np.asarray(y)
            q1 = np.asarray(q1)
            q3 = np.asarray(q3)

            if x.shape[0] != y.shape[0]:
                print(x.shape, y.shape)
                diff = x.shape[0] - y.shape[0] 
                x = x[:-diff]
            
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



# Only makes sense for multi epoch workload
def plot_from_raw_epoch_completion_times(data_dir, detail=None):
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]
    log_files.sort()

    all_metrics = copy.deepcopy(durations)
    all_metrics["sum_computation"] = []
    all_metrics["step_summed"] = []
    all_metrics["sum_loading"] = []
    all_metrics['sum_all_compute_and_load_gpu'] = []
    all_metrics['diff_step_vs_summed'] = []
    all_metrics['time_to_first_forward'] = []

    # DS for plotting data
    plotting_data = {}
    for num_GPU in GPUs_int:
        plotting_data[num_GPU] = {
            batch_num: copy.deepcopy(all_metrics) for batch_num in batches_int
        }
    
    with open(os.path.join(data_dir, "step_analysis.txt"), "w") as outfile:

        outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}\n")

        for log_file in log_files:
            outfile.write(f"{log_file}\n")
            gpu_key = get_num_gpus(log_file)
            batch_key = get_batch_size(log_file)
            # batch_key = get_num_workers(log_file)

            all_times = copy.deepcopy(all_metrics)

            infile = open(log_file, mode='r')
            log = json.load(infile)
            infile.close()

            # Gather all durations for each epoch in a log file
            step_summed = sum_computation = sample_load = sample_preproc = 0
            sum_all_compute_and_load_gpu = sum_loading = 0

            time_to_first_forward = []
            epoch_completion = []
            t_epoch_start = 0
            got_first_forward = False
            epoch_finished = False # just in case


            for line in log:
                if 'metadata' in line and 'epoch_num' in line['metadata'] and line['metadata']['epoch_num'] == 1:
                    continue

                if line['key'] == "epoch_start":
                    epoch = line['metadata']['epoch_num']

                    t_epoch_start = np.datetime64(line['time_ms'])
                    got_first_forward = False
                    epoch_finished = False

                if not epoch_finished and line['key'] == "epoch_stop":
                    epoch_finished = True
                    diff = np.datetime64(line['time_ms']) - t_epoch_start
                    diff = np.timedelta64(diff, 'ms') 
                    diff = diff.astype(np.int64) / 1000.0
                    print(f'Epoch completion time: {diff}s')
                    epoch_completion.append(diff)


                if not got_first_forward and line['key'] == 'model_forward_pass':
                    got_first_forward = True
                    diff = np.datetime64(line['time_ms']) - t_epoch_start
                    diff = np.timedelta64(diff, 'ms') 
                    diff = diff.astype(np.int64) / 1000.0
                    print(f'Time to first fwd: {diff}')
                    time_to_first_forward.append(diff)

                if line['key'] in durations.keys():
                    # Append value to appropriate array
                    value = round(line['value']['duration'] / 1_000_000_000, 3)
                    # value = round(line['value']['duration'], 4)

                    if line['key'] == 'sample_load':
                        sample_load += value
                        continue
                    if line['key'] =='sample_preproc':
                        sample_preproc += value
                        continue

                    if line['key'] == 'step_end':
                        # sum_others = round(sum_others, 3)
                        all_times['step_end'].append(value)
                        all_times['sum_computation'].append(sum_computation)
                        all_times['sum_all_compute_and_load_gpu'].append(sum_all_compute_and_load_gpu)

                        if "sample_load" in all_metrics and "sample_preproc" in all_metrics:
                            all_times['sample_load'].append(sample_load)
                            all_times['sample_preproc'].append(sample_preproc)

                        all_times['step_summed'].append(step_summed)
                        all_times['sum_loading'].append(sum_loading)

                        # print(f"step: {value}\tsum: {sum_others}\tdiff: {round(value - sum_others, 3)}") 
                        # Reset values
                        step_summed = sum_computation = sample_load = sample_preproc = sum_loading = 0
                        sum_all_compute_and_load_gpu = 0
                    else:
                        all_times[line['key']].append(value)

                        if line['key'] != 'all_compute':
                            step_summed += value

                            if line['key'] != 'load_batch_mem':
                                sum_computation += value

                        if line['key'] == 'load_batch_mem' or line['key'] == 'load_batch_gpu':
                            sum_loading += value

                        if line['key'] == 'load_batch_gpu' or line['key'] == 'all_compute':
                            sum_all_compute_and_load_gpu += value


            all_times['time_to_first_forward'] = np.asarray(time_to_first_forward)    
            all_times['epoch_completion'] = np.asarray(epoch_completion)    
            all_times['diff_step_vs_summed'] = np.asarray(all_times['step_end']) - np.asarray(all_times['step_summed'])
            all_times['diff_step_vs_summed'] = all_times['diff_step_vs_summed'].tolist()
            
            for key in all_times:
                print(key)
                mean = stats.mean(all_times[key])
                median = stats.median(all_times[key])
                std = stats.stdev(all_times[key])
                quartiles = stats.quantiles(all_times[key])

                plotting_data[gpu_key][batch_key][key] = {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'q1': quartiles[0],
                    'q3': quartiles[2],
                }
                ROUND = 4
                outfile.write(f"{key:>30}:\t{round(mean, ROUND):>15}\t{round(median, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
            outfile.write("\n")

    print("Plotting step breakdown from raw data for epochs > 1")


    ###############################################################
    # Modify metrics, gpus or batch sizes to plot here
    ###############################################################


    metrics_to_plot_pretty_names = {
        # "epoch_completion": "Time per epoch (>1)",
        "time_to_first_forward": "Time to 1st forward",
        "step_end": "Overall",
        "load_batch_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calc",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
        "cum_loss_fn_calc": "2.5 Cumulative loss",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = [str(b) for b in batches_to_plot]

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5), sharey=True)
    # fig.suptitle(f"{WORKLOAD}{' ' + detail + ' ' if detail else ' '}Step Breakdown (epoch > 1)", fontsize=FONTSIZE)

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
            x = np.asarray(batches_int)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.1)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            if len(batches_to_plot_str[0]) > 3:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            else:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, ha='center', fontsize=FONTSIZE-2)

            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            # ax.set_xlabel("Batch size", fontsize=FONTSIZE)
            # ax.set_ylabel("Time (s)", fontsize=FONTSIZE)
            # ax.legend(fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.01, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

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



    print(f'Printing Eopch completion times plot')
    ## Shorter plot 

    metrics_pretty_names = {
        "epoch_completion": "Epoch time (>1)",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }
    metric = "epoch_completion"

    # Overall plot
    fig, ax = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 6), sharey=True)

    FONTSIZE = 18
    ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

    # plot the metric in the axes
    for gpu_key in GPUs_to_plot:
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
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (0, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch size', fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_epoch_completion.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_from_raw_workers(data_dir, detail=None):
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]
    log_files.sort()

    all_metrics = copy.deepcopy(durations)
    all_metrics["sum_computation"] = []
    all_metrics["step_summed"] = []
    all_metrics["sum_loading"] = []
    all_metrics['sum_all_compute_and_load_gpu'] = []
    all_metrics['diff_step_vs_summed'] = []
    all_metrics['time_to_first_forward'] = []

    # DS for plotting data
    plotting_data = {}
    for num_GPU in GPUs_int:
        plotting_data[num_GPU] = {
            batch_num: copy.deepcopy(all_metrics) for batch_num in batches_int
        }
    
    with open(os.path.join(data_dir, "step_analysis.txt"), "w") as outfile:

        outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}\n")

        for log_file in log_files:
            outfile.write(f"{log_file}\n")
            gpu_key = get_num_gpus(log_file)
            batch_key = get_num_workers(log_file)

            print(f'Found num workers: {batch_key}')

            all_times = copy.deepcopy(all_metrics)

            infile = open(log_file, mode='r')
            log = json.load(infile)
            infile.close()

            # Gather all durations for each epoch in a log file
            step_summed = sum_computation = sample_load = sample_preproc = 0
            sum_all_compute_and_load_gpu = sum_loading = 0

            time_to_first_forward = []
            t_epoch_start = 0
            got_first_forward = False


            for line in log:
                if 'metadata' in line and 'epoch_num' in line['metadata'] and line['metadata']['epoch_num'] == 1:
                    continue

                if line['key'] == "epoch_start":
                    epoch = line['metadata']['epoch_num']

                    t_epoch_start = np.datetime64(line['time_ms'])
                    got_first_forward = False


                if not got_first_forward and line['key'] == 'model_forward_pass':
                    got_first_forward = True
                    diff = np.datetime64(line['time_ms']) - t_epoch_start
                    diff = np.timedelta64(diff, 'ms') 
                    diff = diff.astype(np.int64) / 1000.0
                    print(f'Time to first fwd: {diff}')
                    time_to_first_forward.append(diff)

                if line['key'] in durations.keys():
                    # Append value to appropriate array
                    value = round(line['value']['duration'] / 1_000_000_000, 3)
                    # value = round(line['value']['duration'], 4)

                    if line['key'] == 'sample_load':
                        sample_load += value
                        continue
                    if line['key'] =='sample_preproc':
                        sample_preproc += value
                        continue

                    if line['key'] == 'step_end':
                        # sum_others = round(sum_others, 3)
                        all_times['step_end'].append(value)
                        all_times['sum_computation'].append(sum_computation)
                        all_times['sum_all_compute_and_load_gpu'].append(sum_all_compute_and_load_gpu)

                        if "sample_load" in all_metrics and "sample_preproc" in all_metrics:
                            all_times['sample_load'].append(sample_load)
                            all_times['sample_preproc'].append(sample_preproc)

                        all_times['step_summed'].append(step_summed)
                        all_times['sum_loading'].append(sum_loading)

                        # print(f"step: {value}\tsum: {sum_others}\tdiff: {round(value - sum_others, 3)}") 
                        # Reset values
                        step_summed = sum_computation = sample_load = sample_preproc = sum_loading = 0
                        sum_all_compute_and_load_gpu = 0
                    else:
                        all_times[line['key']].append(value)

                        if line['key'] != 'all_compute':
                            step_summed += value

                            if line['key'] != 'load_batch_mem':
                                sum_computation += value

                        if line['key'] == 'load_batch_mem' or line['key'] == 'load_batch_gpu':
                            sum_loading += value

                        if line['key'] == 'load_batch_gpu' or line['key'] == 'all_compute':
                            sum_all_compute_and_load_gpu += value


            # all_times['time_to_first_forward'] = np.asarray(time_to_first_forward)    
            all_times['diff_step_vs_summed'] = np.asarray(all_times['step_end']) - np.asarray(all_times['step_summed'])
            all_times['diff_step_vs_summed'] = all_times['diff_step_vs_summed'].tolist()
            
            for key in all_times:
                mean = stats.mean(all_times[key])
                median = stats.median(all_times[key])
                std = stats.stdev(all_times[key])
                quartiles = stats.quantiles(all_times[key])

                plotting_data[gpu_key][batch_key][key] = {
                    'mean': mean,
                    'median': median,
                    'std': std,
                    'q1': quartiles[0],
                    'q3': quartiles[2],
                }
                ROUND = 4
                outfile.write(f"{key:>30}:\t{round(mean, ROUND):>15}\t{round(median, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
            outfile.write("\n")

    print("Plotting step breakdown from raw data for epochs > 1")


    ###############################################################
    # Modify metrics, gpus or batch sizes to plot here
    ###############################################################


    # metrics_to_plot_pretty_names = {
    #     "step_end": "Overall step",
    #     # "sample_load": "1.1-Batch load",
    #     # "sample_preproc": "1.2-Sample Preproc (CPU)",
    #     "load_batch_mem": "1-Batch Loading",
    #     "sum_computation": "2-Batch Processing",
    #     # "sum_all_compute_and_load_gpu": "2-(sum all_comp and load gpu)",
    #     "all_compute": "2.0-Compute only",
    #     # "load_batch_gpu": "2.1-Batch to GPU",
    #     "model_forward_pass": "2.2-Forward pass",
    #     "loss_tensor_calc": "2.3-Loss calc",
    #     "model_backward_pass": "2.4-Backward pass",
    #     "model_optim_step": "2.5-Optimizer step",
    #     # "cum_loss_fn_calc": "2.6-Cumulative loss",
    # }
    metrics_to_plot_pretty_names = {
        "time_to_first_forward": "Time to 1st forward",
        "step_end": "Overall",
        "load_batch_mem": "1 Batch Loading",
        "sum_computation": "2 Computation",
        "model_forward_pass": "2.1 Forward pass",
        "loss_tensor_calc": "2.2 Loss calc",
        "model_backward_pass": "2.3 Backward pass",
        "model_optim_step": "2.4 Optimizer step",
        # "cum_loss_fn_calc": "2.5 Cumulative loss",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int
    batches_to_plot_str = [str(b) for b in batches_to_plot]

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5), sharey=True)
    # fig.suptitle(f"{WORKLOAD}{' ' + detail + ' ' if detail else ' '}Step Breakdown (epoch > 1)", fontsize=FONTSIZE)

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

        if i_ax == 2:
            ax.spines['right'].set_visible(False)

        if i_ax >= 3:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left = False)
            # ax.yaxis.tick_params(which="major", visible=False)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:
            x = np.asarray(batches_int)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batches_to_plot ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.1)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            if len(batches_to_plot_str[0]) > 3:
                ax.set_xticks(batches_to_plot, batches_to_plot_str, rotation=-35, ha='center', fontsize=FONTSIZE-2)
                # ax.tick_params(axis='x', labelrotation=45)

            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            # ax.set_xlabel("Batch size", fontsize=FONTSIZE)
            # ax.set_ylabel("Time (s)", fontsize=FONTSIZE)
            # ax.legend(fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.01, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel('Num Workers', fontsize=FONTSIZE)

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
    fig.supxlabel('Num Workers', fontsize=FONTSIZE)

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




def export_per_eval_stats(data_dir):
    print("Exporting per eval stats")
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))]

    for log_file in log_files:
        print(f"Processing {log_file}")

        # Hardcoded number of evals (3)
        per_eval_durations = {
            n + 1: copy.deepcopy(eval_durations) for n in range(0, NUM_EVALS)
        }
        per_eval_stats = {
            n + 1: copy.deepcopy(eval_durations) for n in range(0, NUM_EVALS)
        }

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()

        eval_num = 0

        # Gather all durations for each epoch in a log file
        for line in log:

            if line['key'] == "eval_start":
                eval_num += 1
                continue

            if line['key'] in eval_durations.keys():

                # UNET3D specific
                # Save the size only once, else we will have the same value 4 times
                if line['key'] == 'eval_load_batch_mem' and 'image_shape' in line['value']:
                    # The shape gives the number of pixels, each is of type np.float32 so 4 bytes
                    sample_size = int(np.prod(line['value']['image_shape'])) * 4
                    per_eval_durations[eval_num]['eval_image_sizes'].append(sample_size)

                value = line['value']['duration']
                per_eval_durations[eval_num][line['key']].append(value)

        # Calculate stats for each evaluation in current log file
        for eval_num, data in per_eval_durations.items():
            for metric in eval_durations.keys():
                per_eval_stats[eval_num][metric] = {
                    "mean": int(stats.mean(data[metric])),
                    "stdev": int(stats.pstdev(data[metric])),
                }
        
        outfile_base = os.path.basename(log_file).replace(".json", "")

        ##############################
        # Save eval data
        ##############################
        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, "per_eval")).mkdir(parents=True, exist_ok=True)

        outfile_stats = outfile_base + "_per_eval_stats.json"
        outfile_durations = outfile_base + "_per_eval_durations.json"

        json.dump(per_eval_stats, open(os.path.join(data_dir, "per_eval", outfile_stats), mode="w"), indent=4)
        json.dump(per_eval_durations, open(os.path.join(data_dir, "per_eval", outfile_durations), mode="w"), indent=4)




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

def UNET_export_overall_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, "per_epoch", f) for f in os.listdir(os.path.join(data_dir, "per_epoch")) if re.match(rf'{WORKLOAD}_.*_durations\.json', f)]

    overall_stats = {}
    for num_GPU in GPUs_int:
        overall_stats[num_GPU] = {
            batch_num: copy.deepcopy(durations) for batch_num in batches_int
        }
    
    overall_durations = copy.deepcopy(durations)
    overall_durations["sum_all"] = np.asarray([], dtype=np.int64)
    overall_durations["all_compute_summed"] = np.asarray([], dtype=np.int64)
    overall_durations["total_minus_load"] = np.asarray([], dtype=np.int64)


    for log_file in log_files:

        print(f"Processing {log_file}")
        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()
 
        gpu_key = get_num_gpus(log_file)
        batch_key = get_batch_size(log_file)
        print(f"GPUs: {gpu_key}, batch size: {batch_key}")

        if gpu_key not in GPUs_int or batch_key not in batches_int:
            continue
        
        total_minus_load = np.asarray([], dtype=np.int64)

        # Calculate stats for each epoch in current log file
        for epoch, data in log.items():

            # Used to calculate means over all but first epoch
            if epoch == "1" or epoch == 1:
                print('skipping epoch 1')
                continue

            min_size = get_smallest_common_length(data)

            diff = np.asarray(data["step_end"][:min_size]) - np.asarray(data["load_batch_mem"][:min_size])

            overall_durations["total_minus_load"] = np.append(overall_durations["total_minus_load"], diff)
            total_minus_load = np.append(total_minus_load, diff)

            # Sample load and preproc are per SAMPLE values, while everything else is per BATCH
            # So we will sum up batch_size values of sample load and preproc to obtain comparable values
            if "sample_load" in data and "sample_preproc" in data:

                sample_load = np.asarray(data["sample_load"])
                sample_preproc = np.asarray(data["sample_preproc"])

                pad_size = (min_size * batch_key) - len(sample_load)

                if pad_size >= 0:
                    sample_load = np.pad(sample_load, [0, pad_size])
                    sample_preproc = np.pad(sample_preproc, [0, pad_size])
                else:
                    sample_load = sample_load[:pad_size]
                    sample_preproc = sample_preproc[:pad_size]

                # Reshape groups them into batch_size long subarrays.
                # we then sum over those
                sample_load = sample_load.reshape(-1, batch_key).sum(1)
                sample_preproc = sample_preproc.reshape(-1, batch_key).sum(1)


            sum_all = np.zeros(shape=(min_size))
            all_compute_summed = np.zeros(shape=(min_size))

            for metric in data.keys():

                if metric == "sample_load":
                    overall_stats[gpu_key][batch_key][metric].extend(sample_load.tolist())
                    overall_durations[metric].extend(sample_load)
                    continue
                elif metric == "sample_preproc":
                    overall_stats[gpu_key][batch_key][metric].extend(sample_preproc.tolist())
                    overall_durations[metric].extend(sample_preproc)
                    continue

                if metric != "step_end":
                    # Sum all will not include step_end, sample_load and sample_preproc
                    # so it is the sum of steps 1-7
                    sum_all += np.asarray(data[metric][:min_size])

                    if metric != "load_batch_mem" and metric != 'all_compute': 
                        all_compute_summed += np.asarray(data[metric][:min_size])


                overall_stats[gpu_key][batch_key][metric].extend(data[metric])
                overall_durations[metric].extend(data[metric])

            
            overall_durations["sum_all"] = np.append(overall_durations["sum_all"], sum_all)
            overall_durations["all_compute_summed"] = np.append(overall_durations["all_compute_summed"], all_compute_summed)

        overall_stats[gpu_key][batch_key]["total_minus_load"] = total_minus_load.tolist()
        overall_stats[gpu_key][batch_key]["sum_all"] = sum_all.tolist()
        overall_stats[gpu_key][batch_key]["all_compute_summed"] = all_compute_summed.tolist()

    print("computing actual overall means")

    # Set to 1_000_000_000 for data in ns
    CONVERSION = 1

    outdir = os.path.join(data_dir, "overall")
    pathlib.Path(outdir).mkdir(exist_ok=True)

    actual_overall_stats = copy.deepcopy(durations)

    for metric in overall_durations.keys():
        print(metric)
        quartiles = stats.quantiles(overall_durations[metric])
        actual_overall_stats[metric] = {
            "mean": round(np.asarray(overall_durations[metric]).mean() / CONVERSION, 3),
            "median": round(np.median(np.asarray(overall_durations[metric])) / CONVERSION, 3),
            "stdev": round(np.asarray(overall_durations[metric]).std() / CONVERSION, 3),
            "q1": round(quartiles[0] / CONVERSION, 3),
            "q3": round(quartiles[2] / CONVERSION, 3),
        }


    with open(os.path.join(outdir, f"{WORKLOAD}_epoch_actual_overall.json"), mode="w") as outfile:
        json.dump(actual_overall_stats, outfile, indent=4)

    with open(os.path.join(outdir, f"{WORKLOAD}_epoch_overall_stats.json"), mode="w") as outfile:
        json.dump(overall_stats, outfile, indent=4)
    
    all_exported_metrics = list(durations.keys())
    all_exported_metrics.append("total_minus_load")
    all_exported_metrics.append("sum_all")
    all_exported_metrics.append("all_compute_summed")

    print(all_exported_metrics)
    # Compute and export overall means
    overall_means = {}
    for gpu_key in GPUs_int:
        overall_means[gpu_key] = {}
        for batch_key in batches_int:
            overall_means[gpu_key][batch_key] = copy.deepcopy(durations)

            for metric in all_exported_metrics:
                data = overall_stats[gpu_key][batch_key][metric]
                
                overall_means[gpu_key][batch_key][metric] = {}
                overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / CONVERSION
                overall_means[gpu_key][batch_key][metric]["median"] = stats.median(data) / CONVERSION
                overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / CONVERSION
                quartiles = stats.quantiles(data)
                overall_means[gpu_key][batch_key][metric]["q1"] = quartiles[0] / CONVERSION
                overall_means[gpu_key][batch_key][metric]["q3"] = quartiles[2] / CONVERSION


    with open(os.path.join(outdir, f"{WORKLOAD}_epoch_overall_means.json"), mode="w") as outfile:
        json.dump(overall_means, outfile, indent=4)



def DLRM_export_overall_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, "per_epoch", f) for f in os.listdir(os.path.join(data_dir, "per_epoch")) if re.match(rf'{WORKLOAD}_.*_durations\.json', f)]

    overall_stats = {}
    for num_GPU in GPUs_int:
        overall_stats[num_GPU] = {
            batch_num: copy.deepcopy(durations) for batch_num in batches_int
        }

    overall_durations = copy.deepcopy(durations)
    overall_durations["sum_all"] = np.asarray([], dtype=np.int64)
    overall_durations["sum_all_but_load"] = np.asarray([], dtype=np.int64)
    overall_durations["total_minus_load"] = np.asarray([], dtype=np.int64)

    for log_file in log_files:

        print(f"Processing {log_file}")
        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()
 
        gpu_key = get_num_gpus(log_file)
        batch_key = get_batch_size(log_file)
        print(f"GPUs: {gpu_key}, batch size: {batch_key}")
        
        over_all_epochs_sum_all = np.asarray([], dtype=np.int64)
        over_all_epochs_sum_all_but_load = np.asarray([], dtype=np.int64)
        over_all_epochs_total_minus_load = np.asarray([], dtype=np.int64)

        # Calculate stats for each epoch in current log file
        for epoch, data in log.items():

            min_size = get_smallest_common_length(data)
            
            diff = np.asarray(data["step_end"][:min_size]) - np.asarray(data["load_batch_mem"][:min_size])
            overall_durations["total_minus_load"] = np.append(overall_durations["total_minus_load"], diff)
            over_all_epochs_total_minus_load = np.append(over_all_epochs_total_minus_load, diff)

            sum_all = np.zeros(shape=(min_size))
            sum_all_but_load = np.zeros(shape=(min_size))
            for metric in data.keys():

                if metric != "step_end":
                    # Sum all will not include step_end, sample_load and sample_preproc
                    # so it is the sum of steps 1-7
                    sum_all += np.asarray(data[metric][:min_size])

                    if metric != "load_batch_mem":
                        sum_all_but_load += np.asarray(data[metric][:min_size])

                overall_stats[gpu_key][batch_key][metric].extend(data[metric])
                overall_durations[metric].extend(data[metric])
            
            over_all_epochs_sum_all = np.append(over_all_epochs_sum_all, sum_all)
            over_all_epochs_sum_all_but_load = np.append(over_all_epochs_sum_all_but_load, sum_all_but_load)

            overall_durations["sum_all"] = np.append(overall_durations["sum_all"], sum_all)
            overall_durations["sum_all_but_load"] = np.append(overall_durations["sum_all_but_load"], sum_all_but_load)

        overall_stats[gpu_key][batch_key]["sum_all"] = over_all_epochs_sum_all.tolist()
        overall_stats[gpu_key][batch_key]["sum_all_but_load"] = over_all_epochs_sum_all_but_load.tolist()
        overall_stats[gpu_key][batch_key]["total_minus_load"] = over_all_epochs_total_minus_load.tolist()


    print("computing actual overall means")

    actual_overall_stats = copy.deepcopy(durations)

    for metric in overall_durations.keys():
        quartiles = stats.quantiles(overall_durations[metric])
        actual_overall_stats[metric] = {
            "mean": round(np.asarray(overall_durations[metric]).mean() / 1_000_000_000, 3),
            "stdev": round(np.asarray(overall_durations[metric]).std() / 1_000_000_000, 3),
            "q1": round(quartiles[0] / 1_000_000_000, 3),
            "q3": round(quartiles[2] / 1_000_000_000, 3),
        }

    pathlib.Path(os.path.join(data_dir,"overall")).mkdir(exist_ok=True, parents=True)

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_actual_overall.json"), mode="w") as outfile:
        json.dump(actual_overall_stats, outfile, indent=4)

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_stats.json"), mode="w") as outfile:
        json.dump(overall_stats, outfile, indent=4)
    

    all_metrics_to_mean = list(durations.keys()) + ["total_minus_load"]
    # Compute and export overall means
    overall_means = {}
    for gpu_key in GPUs_int:
        overall_means[gpu_key] = {}
        for batch_key in batches_int:
            overall_means[gpu_key][batch_key] = copy.deepcopy(durations)

            for metric in all_metrics_to_mean:
                data = overall_stats[gpu_key][batch_key][metric]
                
                overall_means[gpu_key][batch_key][metric] = {}
                overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["median"] = stats.median(data) / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / 1_000_000_000

                quartiles = stats.quantiles(data)
                overall_means[gpu_key][batch_key][metric]["q1"] = quartiles[0] / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["q3"] = quartiles[2] / 1_000_000_000

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_means.json"), mode="w") as outfile:
        json.dump(overall_means, outfile, indent=4)


def export_overall_eval_stats(data_dir):

    log_files = [os.path.join(data_dir, "per_eval", f) for f in os.listdir(os.path.join(data_dir, "per_eval")) if re.match(rf'{WORKLOAD}_.*_durations\.json', f)]

    eval_overall_stats = {}
    for num_GPU in GPUs_int:
        eval_overall_stats[num_GPU] = {
            batch_num: copy.deepcopy(eval_durations) for batch_num in batches_int
        }
    
    eval_overall_durations = copy.deepcopy(eval_durations)
    eval_overall_durations["total_minus_load"] = np.asarray([], dtype=np.int64)
    eval_overall_durations["sum_all_except_load"] = np.asarray([], dtype=np.int64)


    for log_file in log_files:

        print(f"Processing {log_file}")
        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()
 
        gpu_key = get_num_gpus(log_file)
        batch_key = get_batch_size(log_file)
        print(f"GPUs: {gpu_key}, batch size: {batch_key}")
        
        # Calculate stats for each eval in current log file
        for eval_num, data in log.items():
            # Used to calculate means over all but first eval
            if eval_num == 1:
                continue

            min_size = get_smallest_common_length(data)
            # print(min_size)
            
            diff = np.asarray(data["eval_step_end"][:min_size]) - np.asarray(data["eval_load_batch_mem"][:min_size])
            eval_overall_durations["total_minus_load"] = np.append(eval_overall_durations["total_minus_load"], diff)

            sum_all_except_load = np.zeros(shape=(min_size))
            for metric in data.keys():
                # print(f"{metric} {len(data[metric])}")
                if metric != "eval_step_end" and metric != "eval_load_batch_mem":
                    sum_all_except_load += np.asarray(data[metric][:min_size])

                eval_overall_stats[gpu_key][batch_key][metric].extend(data[metric])
                eval_overall_durations[metric].extend(data[metric])
            
            eval_overall_durations["sum_all_except_load"] = np.append(eval_overall_durations["sum_all_except_load"], sum_all_except_load)
            # print(overall_durations["sum_all_except_load"].shape)

    print("computing actual overall means")
    # Compute actual OVERALL - across 4 and 8 GPUs and all batch sizes

    eval_actual_overall_stats = copy.deepcopy(eval_durations)

    # Set to 1_000_000_000 for data in ns
    CONVERSION = 1

    for metric in eval_overall_durations.keys():
        # These are np arrays so we process them differently
        if metric in ["eval_image_sizes"]:
            eval_actual_overall_stats[metric] = {
                "mean": round(stats.mean(eval_overall_durations[metric])),
                "stdev": round(stats.pstdev(eval_overall_durations[metric]))
            }    
        else:
            eval_actual_overall_stats[metric] = {
                "mean": round(np.asarray(eval_overall_durations[metric]).mean() / CONVERSION, 3),
                "stdev": round(np.asarray(eval_overall_durations[metric]).std() / CONVERSION, 3)
            }

    with open(os.path.join(data_dir, f"{WORKLOAD}_eval_actual_overall.json"), mode="w") as outfile:
        json.dump(eval_actual_overall_stats, outfile, indent=4)

    with open(os.path.join(data_dir, f"{WORKLOAD}_eval_overall_stats.json"), mode="w") as outfile:
        json.dump(eval_overall_stats, outfile, indent=4)
    
    # Compute and export overall means
    eval_overall_means = {}
    for gpu_key in GPUs_int:
        eval_overall_means[gpu_key] = {}
        for batch_key in batches_int:
            # Create the DS for the overall eval means
            eval_overall_means[gpu_key][batch_key] = copy.deepcopy(eval_durations)

            for metric in eval_durations.keys():
                data = eval_overall_stats[gpu_key][batch_key][metric]
                
                eval_overall_means[gpu_key][batch_key][metric] = {}
                eval_overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / CONVERSION
                eval_overall_means[gpu_key][batch_key][metric]["median"] = stats.median(data) / CONVERSION
                eval_overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / CONVERSION

                quartiles = stats.quantiles(data)
                eval_overall_means[gpu_key][batch_key][metric]["q1"] = quartiles[0] / CONVERSION
                eval_overall_means[gpu_key][batch_key][metric]["q3"] = quartiles[2] / CONVERSION

    outdir = os.path.join(data_dir, "overall")
    pathlib.Path(outdir).mkdir(exist_ok=True)

    with open(os.path.join(outdir, f"{WORKLOAD}_eval_overall_means.json"), mode="w") as outfile:
        json.dump(eval_overall_means, outfile, indent=4)


def plot_overall_step_time_curves(data_dir):
    
    with open(os.path.join(data_dir, "overall", "overall_means.json"), mode="r") as infile:
        overall_means = json.load(infile)

    # Overall plot
    fig, ax = plt.subplots(nrows=1, ncols=1, layout="constrained", figsize=(6,5))
    fig.suptitle("Average Step Times (epochs > 1) ")

    for gpu_key in GPUs_str:
        x = np.asarray(batches_int)

        y = [ overall_means[gpu_key][batch]["step_end"]["mean"] for batch in batches_str ]
        y = np.asarray(y)

        std = [ overall_means[gpu_key][batch]["step_end"]["stdev"] for batch in batches_str ]
        std = np.asarray(std)
        
        ax.plot(x, y, label=f"{gpu_key} GPUs")
        ax.fill_between(x, y-std, y+std, alpha=0.35)

    ax.legend()

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time taken (s)")
    # Create output directory if it doesn't exist
    pathlib.Path(os.path.join(data_dir, "plots")).mkdir(parents=True, exist_ok=True)

    figure_filename = os.path.join(data_dir, "plots", "overall_step_times.png")

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



# Individual phases plots
def UNET_plot_epoch_individual_time_curves(data_dir, detail=None):
    print("Plotting relative time curves for epochs")

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_means.json"), mode="r") as infile:
        overall_means = json.load(infile)

    metrics_pretty_names = {
        "step_end": "Overall step",
        # "sample_load": "1.1-Batch load",
        # "sample_preproc": "1.2-Sample Preproc (CPU)",
        "load_batch_mem": "1-Batch load",
        "total_minus_load": "2-Batch Processing",
        "all_compute_summed": "2-Batch Processing (sum)",
        "load_batch_gpu": "2.1-Batch to GPU",
        "model_forward_pass": "2.2-Forward pass",
        "loss_tensor_calc": "2.3-Loss calc",
        "model_backward_pass": "2.4-Backward pass",
        "model_optim_step": "2.5-Optimizer step",
        "cum_loss_fn_calc": "2.6-Cumulative loss",
    }
    durations = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(durations.keys()), layout="constrained", figsize=(3.1 * len(durations.keys()), 8), sharey=True)
    fig.suptitle(f"{WORKLOAD}{' ' + detail if detail else ''} Median Time per Step Component")

    i_ax = -1
    for metric in durations.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric])

        batches_int2 = batches_int
        batches_str2 = batches_str

        # plot the metric in the axes
        for gpu_key in GPUs_str:
            x = np.asarray(batches_int2)

            y = [ overall_means[gpu_key][batch][metric]["median"] for batch in batches_str2 ]
            y = np.asarray(y)

            # std = [ overall_means[gpu_key][batch][metric]["stdev"] for batch in batches_str2]
            # std = np.asarray(std)

            q1 = [ overall_means[gpu_key][batch][metric]["q1"] for batch in batches_str2 ]
            q1 = np.asarray(q1)

            q3 = [ overall_means[gpu_key][batch][metric]["q3"] for batch in batches_str2 ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            ax.fill_between(x, q1, q3, alpha=0.05)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            if len(batches_str[0]) > 3:
                ax.set_xticks(batches_int2, batches_str2, rotation=-45, ha='center', fontsize=7)
                # ax.tick_params(axis='x', labelrotation=45)

            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            ax.legend()

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "step_breakdown")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_epoch_breakdown.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def UNET_plot_epoch_step_breakdown_paper(data_dir, detail=None):
    print("Plotting relative time curves for epochs (paper)")

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_means.json"), mode="r") as infile:
        overall_means = json.load(infile)

    metrics_pretty_names = {
        "step_end": "Overall Step",
        "load_batch_mem": "Loading",
        "all_compute": "Computation",
    }
    durations = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(durations.keys()), layout="constrained", figsize=(3.1 * len(durations.keys()), 8), sharey=True)

    FONTSIZE = 18
    i_ax = -1
    for metric in durations.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        batches_int2 = batches_int
        batches_str2 = batches_str

        # plot the metric in the axes
        for gpu_key in GPUs_str:
            x = np.asarray(batches_int2)

            y = [ overall_means[gpu_key][batch][metric]["median"] for batch in batches_str2 ]
            y = np.asarray(y)

            q1 = [ overall_means[gpu_key][batch][metric]["q1"] for batch in batches_str2 ]
            q1 = np.asarray(q1)

            q3 = [ overall_means[gpu_key][batch][metric]["q3"] for batch in batches_str2 ]
            q3 = np.asarray(q3)

            # print(f"GPU {gpu_key} {metric}:\n\t{x}\n\t{y}\n\t{q1}\n\t{q3}")
            
            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            # ax.fill_between(x, y-std, y+std, alpha=0.15)
            ax.fill_between(x, q1, q3, alpha=0.05)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            if len(batches_str[0]) > 3:
                ax.set_xticks(batches_int2, batches_str2, rotation=-45, ha='center', fontsize=FONTSIZE)
                # ax.tick_params(axis='x', labelrotation=45)

            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.set_xlabel("Batch size", fontsize=FONTSIZE)
            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE)

            # ax.legend()

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (0, -0.045, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supylabel("Time taken (s)", fontsize=FONTSIZE)
    fig.supxlabel('Batch size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "step_breakdown")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_epoch_breakdown_paper.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def DLRM_plot_epoch_individual_time_curves(data_dir, sharey=True):
    print("Plotting relative time curves for epochs")

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_means.json"), mode="r") as infile:
        overall_means = json.load(infile)

    metrics_pretty_names = {
        "step_end": "Overall step",
        "load_batch_mem": "1-Batch load to mem",
        "total_minus_load": "2-Batch processing",
        "model_forward_pass": "2.1-Forward pass",
        "loss_tensor_calc": "2.2-Loss calc",
        "model_backward_pass": "2.3-Backward pass",
        "model_optim_step": "2.4-Update gradients",
        # "sum_all_but_load": "All batch processing (add)",
    }
    durations = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(durations.keys()), layout="constrained", figsize=(3.1 * len(durations.keys()), 8), sharey=sharey)
    fig.suptitle(f"{WORKLOAD} Average Time per Step Component")

    i_ax = -1
    for metric in durations.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric])

        # # Remove the DLRM 262k batch size if desired
        batches_int2 = batches_int[:-1]
        batches_str2 = batches_str[:-1]
        # batches_int2 = batches_int
        # batches_str2 = batches_str

        # plot the metric in the axes
        for gpu_key in GPUs_str:
            x = np.asarray(batches_int2)

            y = [ overall_means[gpu_key][batch][metric]["mean"] for batch in batches_str2 ]
            y = np.asarray(y)

            std = [ overall_means[gpu_key][batch][metric]["stdev"] for batch in batches_str2]
            std = np.asarray(std)

            q1 = [ overall_means[gpu_key][batch][metric]["q1"] for batch in batches_str2 ]
            q1 = np.asarray(q1)

            q3 = [ overall_means[gpu_key][batch][metric]["q3"] for batch in batches_str2 ]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            # ax.fill_between(x, y-std, y+std, alpha=0.15)
            ax.fill_between(x, y-q1, y+q3, alpha=0.15)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            if len(batches_str[0]) > 3:
                ax.set_xticks(batches_int2, batches_str2, rotation=-45, ha='center', fontsize=7)
                # ax.tick_params(axis='x', labelrotation=45)

            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            ax.legend()

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "step_breakdown")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}_epoch_breakdown.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def plot_epoch_violin(data_dir, sharey=True):
    
    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_stats.json"), mode="r") as infile:
        all_values = json.load(infile)

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(durations.keys()), layout="constrained", figsize=(3.1 * len(durations.keys()), 8), sharey=sharey)
    fig.suptitle("Average Time per Step Phase (epochs > 1) ")

    def set_axis_style(ax, labels):
        ax.xaxis.set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Sample name')

    i_ax = -1
    for metric in durations.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric])

        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((patches.Patch(color=color), label))

        # plot the metric in the axes
        for gpu_key in GPUs_str:
            x = np.asarray(batches_int)

            for i, batch in enumerate(batches_str):
                y = all_values[gpu_key][batch][metric][1:]
                y = np.asarray(y) / 1_000_000_000
                ax.violinplot(y, positions=[i], showextrema=False, showmeans=True)
                # add_label(, f"{gpu_key} GPUs")    
            
            # ax.plot(x, y, label=f"{gpu_key} GPUs")

            # ax.set_xscale('log', base=2)
            # ax.set_yscale('log', base=10)
            # for large batch sizes
            # ax.set_xticks(batches_int, batches_str, rotation=45, ha='left')

            # set style for the axes
            set_axis_style(ax, batches_str)
            ax.tick_params(axis='x', labelrotation=45)

            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            # ax.legend()
            # ax.legend(*zip(*labels), loc=2)

    # Create output directory if it doesn't exist
    pathlib.Path(os.path.join(data_dir, "plots")).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}_epoch_violin.png" if sharey else f"{WORKLOAD}_epoch_violin.png"
    figure_filename = os.path.join(data_dir, "plots", filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_histograms(data_dir):

    log_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if re.match(f'.*_instrumented\.json', f)]


    for log_file in log_files:

        per_epoch_durations = {
            n: copy.deepcopy(durations) for n in range(1, 11)
        }

        per_epoch_stats = {
            n: copy.deepcopy(durations) for n in range(1, 11)
        }

        print(f"Processing {log_file}")

        fig = plt.figure(constrained_layout=True, figsize=(30,20))
        fig.suptitle(os.path.basename(log_file).replace(".json", "") + " training phases time distribution")

        # create 1 subfig per epoch
        subfigs = fig.subfigures(nrows=10, ncols=1)

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()

        # Gather all durations for each epoch in a log file
        for line in log:
            # keys: time_ms, event_type, key, value, metadata
            if line['key'] == "epoch_start":
                epoch = line['metadata']['epoch_num']
                continue

            if line['key'] in durations.keys():
                # Append value to appropriate array
                value = line['value']['duration']
                # convert value from nanoseconds to milliseconds
                value = round(value / 1_000_000, 3)
                per_epoch_durations[epoch][line['key']].append(value)
        

        # Calculate stats for each epoch in current log file
        for epoch, data in per_epoch_durations.items():
            i_row =  epoch - 1
            # Get the row subfigure for current epoch
            subfig = subfigs[i_row]
            subfig.suptitle(f'Epoch {epoch}')

            # create subfig axes
            axes = subfig.subplots(nrows=1, ncols=len(durations.keys()))
            
            i_ax = -1
            for metric in durations.keys():
                i_ax += 1
                # Get the correct axis
                ax = axes[i_ax]
                ax.set_title(f'{metrics_pretty_names[metric]} (ms)')

                x = np.asarray(data[metric])
                ax.hist(x, bins=25)
                ax.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)

                props = dict(facecolor='white', alpha=0.5, linewidth=0)

                textstr = f"mean: {round(x.mean(), 2)}\nstd: {round(x.std(), 2)}"

                # place a text box in upper left in axes coords
                ax.text(0.70, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)

                per_epoch_stats[epoch][metric] = {
                    "mean": int(stats.mean(data[metric])),
                    "stdev": int(stats.pstdev(data[metric])),
                }
        
        outfile_base = os.path.basename(log_file).replace("_instrumented", "").replace(".json", "")

        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, "histograms")).mkdir(parents=True, exist_ok=True)

        figure_filename = outfile_base + ".png"
        figure_filename = os.path.join(data_dir, "histograms", figure_filename)

        plt.savefig(figure_filename, format="png", dpi=450)
        # Clear the current axes.
        plt.cla() 
        # Closes all the figure windows.
        plt.close('all')   
        plt.close(fig)


# Individual phases plots
def plot_latency_histograms(data_dir):
    print("Plotting latency histograms")

    metrics_pretty_names = {
        "step_end": "Overall step",
        "all_compute": "1-Batch loaded in memory",
    }
    durations = { metric: [] for metric in metrics_pretty_names }

    output_dir = os.path.join(data_dir, f"histograms")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_epoch_overall_stats.json"), mode="r") as infile:
        all_values = json.load(infile)

    # fig = plt.figure(layout="constrained", figsize=(3.1 * len(durations.keys()), 2 * len(batches_str)))
    # fig.suptitle("Distributions of time taken in each part of a training step")

    i_row = -1

    colors = ['blue', 'orange', 'green', 'red']

        # sharey=True, sharex=True
    # create 1 subfig per epoch
    # subfigs = fig.subfigures(nrows=len(durations.keys()), ncols=1)
    # subfigs = fig.subfigures(nrows=len(durations.keys()), ncols=len(batches_str))

    for metric in durations.keys():
        i_row += 1

        # subfig = subfigs[i_row]
        # subfig.suptitle(metrics_pretty_names[metric])

        # axes = subfig.subplots(nrows=1, ncols=len(batches_str))

        # i_ax = -1
        for batch in batches_str:

            fig, ax = plt.subplots(layout="constrained", figsize=(8, 8))
            fig.suptitle(f"{WORKLOAD}: {metric} latencies distribution")
            # i_ax += 1
            # ax = axes[i_ax]
            ax.set_title(f'Batch size {batch}')
            
            all_data = [ all_values[gpu_key][batch][metric][1:] for gpu_key in GPUs_str ]
            all_data = [np.asarray(x) / 1_000_000_000 for x in all_data]
            
            ax.hist(all_data, bins=100, density=True, histtype='step', fill=True, color=colors, alpha=0.25, label=GPUs_str)

            ax.set_xlabel("Time taken (s)", fontsize=9)
            ax.legend()

            filename = f"{WORKLOAD}_batch{batch}_epoch_{metric}.png"
            figure_filename = os.path.join(output_dir, filename)

            plt.savefig(figure_filename, format="png", dpi=450)
            # Clear the current axes.
            plt.cla() 
            # Closes all the figure windows.
            plt.close('all')   
            plt.close(fig)


# Individual phases plots
def plot_eval_individual_time_curves(data_dir, detail=None):
    print("Plotting relative times curves for evals")
    with open(os.path.join(data_dir, "overall", f"{WORKLOAD}_eval_overall_means.json"), mode="r") as infile:
        eval_overall_means = json.load(infile)

    # Overall plot

    # UNET3D we skip a key
    # num_subplots = len(eval_durations.keys()) - 1
    
    # DLRM plot them all
    num_subplots = len(eval_durations.keys())

    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, layout="constrained", figsize=(3.1 * num_subplots, 8), sharey=True)
    fig.suptitle(f"{WORKLOAD}{' ' + detail if detail else ''} Median Time per Evaluation Step Phase (Eval > 1) ")

    i_ax = -1
    for metric in eval_durations.keys():
        # Don't plot image sizes
        if metric == "eval_image_sizes":
            continue
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(eval_metrics_pretty_names[metric])

        # plot the metric in the axes
        for gpu_key in GPUs_str:
            x = np.asarray(batches_int)

            y = [ eval_overall_means[gpu_key][batch][metric]["median"] for batch in batches_str ]
            y = np.asarray(y)

            # std = [ eval_overall_means[gpu_key][batch][metric]["stdev"] for batch in batches_str]
            # std = np.asarray(std)

            q1 = [ eval_overall_means[gpu_key][batch][metric]["q1"] for batch in batches_str]
            q1 = np.asarray(q1)

            q3 = [ eval_overall_means[gpu_key][batch][metric]["q3"] for batch in batches_str]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            # ax.fill_between(x, y-std, y+std, alpha=0.15)
            ax.fill_between(x, q1, q3, alpha=0.05)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            # ax.set_xticks(batches_int, batches_str, rotation=45, ha='left')
            if len(batches_str[0]) > 3:
                ax.tick_params(axis='x', labelrotation=45)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            ax.legend()

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "step_breakdown")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}{'_' + detail if detail else ''}_eval_breakdown.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def plot_eval_individual_time_curves_by_image_size(data_dir, sharey=True):
    """
    For UNET3D analysis, where image sizes are different
    """
    
    with open(os.path.join(data_dir, f"{WORKLOAD}_eval_overall_stats.json"), mode="r") as infile:
        eval_overall_stats = json.load(infile)

    eval_all_points = copy.deepcopy(eval_durations)

    for gpu_key in GPUs_str:
        for batch_key in batches_str:
            for metric in eval_durations.keys():
                data = eval_overall_stats[gpu_key][batch_key][metric]
                # Gather ALL data points in here
                eval_all_points[metric].extend(data)

    # Overall plot
    num_subplots = len(eval_durations.keys()) - 1
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, layout="constrained", figsize=(3.1 * num_subplots, 8), sharey=sharey)
    fig.suptitle("Eval Phase Time vs Image Size (Eval > 1)")

    # Same X axis for all
    x = np.asarray(eval_all_points["eval_image_sizes"])
    x = x / 1e6

    i_ax = -1
    for metric in eval_durations.keys():
        # Don't plot image sizes
        if metric == "eval_image_sizes":
            continue
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(eval_metrics_pretty_names[metric])

        y = np.asarray(eval_all_points[metric])
        y = y / 1_000_000_000
            
        ax.scatter(x, y)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Image size (MB)", fontsize=10)
        ax.set_ylabel("Time taken (s)", fontsize=10)
        # ax.legend()

    # Create output directory if it doesn't exist
    pathlib.Path(os.path.join(data_dir, "plots")).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}_eval_indiv_times_img_size_sharey.png" if sharey else f"{WORKLOAD}_eval_indiv_times_img_size.png"
    figure_filename = os.path.join(data_dir, "plots", filename)

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

    plot_from_raw(args.data_dir, args.workload, args.name)

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
