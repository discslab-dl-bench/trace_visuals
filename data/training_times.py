import os
import re
import json
import copy
import pathlib
import argparse
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data dictionary that will hold duration values for each epoch
durations = {
    "step_end": [],
    "load_batch_mem": [],
    "sample_load": [],
    "sample_preproc": [],
    "load_batch_gpu": [],
    "model_forward_pass": [],
    "loss_tensor_calc": [],
    "model_backward_pass": [],
    "model_optim_step": [],
    "cum_loss_fn_calc": [],
}

metrics_pretty_names = {
    "step_end": "Overall step",
    "load_batch_mem": "1-Batch loaded in memory",
    "sample_load": "1.1-Sample loaded",
    "sample_preproc": "1.2-Sample preprocessed",
    "load_batch_gpu": "2-Batch loaded to GPU",
    "model_forward_pass": "3-Forward pass",
    "loss_tensor_calc": "4-Loss calculation",
    "model_backward_pass": "5-Backward pass",
    "model_optim_step": "6-Optimizer step",
    "cum_loss_fn_calc": "7-Cumulative loss",
}

eval_durations = {
    "eval_step_end": [],
    "eval_image_sizes": [],
    "eval_load_batch_mem": [],
    "eval_load_batch_gpu": [],
    "eval_sliding_window": [],
    "eval_loss_and_score_fn": [],
}

eval_metrics_pretty_names = {
    "eval_step_end": "Overall eval step",
    "eval_load_batch_mem": "1-Batch loaded in memory",
    "eval_load_batch_gpu": "2-Batch loaded to GPU",
    "eval_sliding_window": "3-Sliding window calc",
    "eval_loss_and_score_fn": "4-Loss and score calc",
    "eval_image_sizes": "Image Sizes"
}

GPUs_int = [2,4,6,8]
GPUs_str = [str(g) for g in GPUs_int]
batches_int = [1, 2, 3, 4, 5]
batches_str = [ str(b) for b in batches_int ]


def export_per_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data')) if re.match(f'.*_instrumented\.json', f)]

    for log_file in log_files:
        print(f"Processing {log_file}")

        # Hardcoded number of epochs (10)
        per_epoch_durations = {
            n: copy.deepcopy(durations) for n in range(1, 11)
        }

        per_epoch_stats = {
            n: copy.deepcopy(durations) for n in range(1, 11)
        }

        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()

        # Gather all durations for each epoch in a log file
        for line in log:

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

        outfile_base = os.path.basename(log_file).replace("_instrumented", "").replace(".json", "")

        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, "per_epoch")).mkdir(parents=True, exist_ok=True)

        outfile_stats = outfile_base + "_per_epoch_stats.json"
        outfile_durations = outfile_base + "_per_epoch_durations.json"

        json.dump(per_epoch_stats, open(os.path.join(data_dir, "per_epoch", outfile_stats), mode="w"), indent=4)
        json.dump(per_epoch_durations, open(os.path.join(data_dir, "per_epoch", outfile_durations), mode="w"), indent=4)
        

def export_per_eval_stats(data_dir):
    print("Exporting per eval stats")
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data')) if re.match(f'.*_instrumented\.json', f)]

    print(log_files)

    for log_file in log_files:
        print(f"Processing {log_file}")

        # Hardcoded number of evals (3)
        per_eval_durations = {
            n: copy.deepcopy(eval_durations) for n in range(1, 4)
        }
        per_eval_stats = {
            n: copy.deepcopy(eval_durations) for n in range(1, 4)
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
                # Save the size only once, else we will have the same value 4 times
                if line['key'] == 'eval_load_batch_mem':
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
        
        outfile_base = os.path.basename(log_file).replace("_instrumented", "").replace(".json", "")

        ##############################
        # Save eval data
        ##############################
        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, "per_eval")).mkdir(parents=True, exist_ok=True)

        outfile_stats = outfile_base + "_per_eval_stats.json"
        outfile_durations = outfile_base + "_per_eval_durations.json"

        json.dump(per_eval_stats, open(os.path.join(data_dir, "per_eval", outfile_stats), mode="w"), indent=4)
        json.dump(per_eval_durations, open(os.path.join(data_dir, "per_eval", outfile_durations), mode="w"), indent=4)


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


def get_smallest_common_length(data):
    l = float("inf")
    for k,v in data.items():
        if len(v) != l:
            l = min(l, len(v))
    return l


def export_overall_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, "per_epoch", f) for f in os.listdir(os.path.join(data_dir, "per_epoch")) if re.match(r'.*_durations\.json', f)]

    overall_stats = {}
    for num_GPU in GPUs_int:
        overall_stats[num_GPU] = {
            batch_num: copy.deepcopy(durations) for batch_num in batches_int
        }
    
    overall_durations = copy.deepcopy(durations)
    overall_durations["sum_all"] = np.asarray([], dtype=np.int64)
    overall_durations["sum_all_except_load_sample"] = np.asarray([], dtype=np.int64)
    overall_durations["sum_all_except_load_and_cum_loss"] = np.asarray([], dtype=np.int64)
    overall_durations["total_minus_load"] = np.asarray([], dtype=np.int64)
    overall_durations["total_minus_load_incl_preproc"] = np.asarray([], dtype=np.int64)
    overall_durations["total_minus_load_minus_cum_loss_incl_preproc"] = np.asarray([], dtype=np.int64)


    for log_file in log_files:

        print(f"Processing {log_file}")
        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()
 
        res = re.search(r'.*([0-9])GPU.*', log_file)
        gpu_key = int(res.group(1))
        batch_key = int(os.path.basename(log_file).split("_")[2].replace("batch", ""))
        print(f"GPUs: {gpu_key}, batch size: {batch_key}")
        
        # Calculate stats for each epoch in current log file
        for epoch, data in log.items():
            # Used to calculate means over all but first epoch
            if epoch == 1:
                continue

            min_size = get_smallest_common_length(data)
            
            diff = np.asarray(data["step_end"][:min_size]) - np.asarray(data["load_batch_mem"][:min_size])
            overall_durations["total_minus_load"] = np.append(overall_durations["total_minus_load"], diff)

            # Sample load and preproc are per SAMPLE values, while everything else is per BATCH
            # So we will sum up batch_size values of sample load and preproc to obtain comparable values
            sample_load = np.asarray(data["sample_load"])
            sample_preproc = np.asarray(data["sample_preproc"])

            pad_size = (min_size * batch_key) - len(sample_load)

            sample_load = np.pad(sample_load, [0, pad_size])
            sample_preproc = np.pad(sample_preproc, [0, pad_size])

            # Reshape groups them into batch_size long subarrays.
            # we then sum over those
            sample_load = sample_load.reshape(-1, batch_key).sum(1)
            sample_preproc = sample_preproc.reshape(-1, batch_key).sum(1)

            # Add back the time taken to preprocess each sample
            diff += sample_preproc
            overall_durations["total_minus_load_incl_preproc"] = np.append(overall_durations["total_minus_load_incl_preproc"], diff)

            # subtract cum_loss from diff to obtain total - load - cum_loss + preproc
            diff -= np.asarray(data["cum_loss_fn_calc"][:min_size])
            overall_durations["total_minus_load_minus_cum_loss_incl_preproc"] = np.append(overall_durations["total_minus_load_minus_cum_loss_incl_preproc"], diff)

            sum_all = np.zeros(shape=(min_size))
            sum_all_except_load_sample = np.zeros(shape=(min_size))
            sum_all_except_load_and_cum_loss = np.zeros(shape=(min_size))
            for metric in data.keys():

                if metric == "sample_load":
                    overall_stats[gpu_key][batch_key][metric].extend(sample_load.tolist())
                    overall_durations[metric].extend(sample_load)
                    continue
                elif metric == "sample_preproc":
                    overall_stats[gpu_key][batch_key][metric].extend(sample_preproc.tolist())
                    overall_durations[metric].extend(sample_preproc)
                    sum_all_except_load_sample += np.asarray(sample_preproc)
                    sum_all_except_load_and_cum_loss += np.asarray(sample_preproc)
                    continue

                if metric != "step_end":
                    # Sum all will not include step_end, sample_load and sample_preproc
                    # so it is the sum of steps 1-7
                    sum_all += np.asarray(data[metric][:min_size])
                    
                    if metric != "load_batch_mem":
                        sum_all_except_load_sample += np.asarray(data[metric][:min_size])

                        if metric != "cum_loss_fn_calc":
                            sum_all_except_load_and_cum_loss += np.asarray(data[metric][:min_size])

                overall_stats[gpu_key][batch_key][metric].extend(data[metric])
                overall_durations[metric].extend(data[metric])
            
            overall_durations["sum_all"] = np.append(overall_durations["sum_all"], sum_all)
            overall_durations["sum_all_except_load_sample"] = np.append(overall_durations["sum_all_except_load_sample"], sum_all_except_load_sample)
            overall_durations["sum_all_except_load_and_cum_loss"] = np.append(overall_durations["sum_all_except_load_and_cum_loss"], sum_all_except_load_and_cum_loss)

    print("computing actual overall means")

    actual_overall_stats = copy.deepcopy(durations)

    for metric in overall_durations.keys():
        actual_overall_stats[metric] = {
            "mean": round(np.asarray(overall_durations[metric]).mean() / 1_000_000_000, 3),
            "stdev": round(np.asarray(overall_durations[metric]).std() / 1_000_000_000, 3)
        }

    with open(os.path.join(data_dir, "epoch_actual_overall.json"), mode="w") as outfile:
        json.dump(actual_overall_stats, outfile, indent=4)

    with open(os.path.join(data_dir, "epoch_overall_stats.json"), mode="w") as outfile:
        json.dump(overall_stats, outfile, indent=4)
    
    # Compute and export overall means
    overall_means = {}
    for gpu_key in GPUs_int:
        overall_means[gpu_key] = {}
        for batch_key in batches_int:
            overall_means[gpu_key][batch_key] = copy.deepcopy(durations)

            for metric in durations.keys():
                data = overall_stats[gpu_key][batch_key][metric]
                
                overall_means[gpu_key][batch_key][metric] = {}
                overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / 1_000_000_000

    with open(os.path.join(data_dir, "epoch_overall_means.json"), mode="w") as outfile:
        json.dump(overall_means, outfile, indent=4)


def export_overall_eval_stats(data_dir):

    log_files = [os.path.join(data_dir, "per_eval", f) for f in os.listdir(os.path.join(data_dir, "per_eval")) if re.match(r'.*_durations\.json', f)]

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
 
        res = re.search(r'.*([0-9])GPU.*', log_file)
        gpu_key = int(res.group(1))

        batch_key = int(os.path.basename(log_file).split("_")[2].replace("batch", ""))
        print(f"GPUs: {gpu_key}, batch size: {batch_key}")
        
        # Calculate stats for each eval in current log file
        for eval_num, data in log.items():
            # Used to calculate means over all but first epoch
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

    for metric in eval_overall_durations.keys():
        # These are np arrays so we process them differently
        if metric in ["eval_image_sizes"]:
            eval_actual_overall_stats[metric] = {
                "mean": round(stats.mean(eval_overall_durations[metric])),
                "stdev": round(stats.pstdev(eval_overall_durations[metric]))
            }    
        else:
            eval_actual_overall_stats[metric] = {
                "mean": round(np.asarray(eval_overall_durations[metric]).mean() / 1_000_000_000, 3),
                "stdev": round(np.asarray(eval_overall_durations[metric]).std() / 1_000_000_000, 3)
            }

    with open(os.path.join(data_dir, "eval_actual_overall.json"), mode="w") as outfile:
        json.dump(eval_actual_overall_stats, outfile, indent=4)

    with open(os.path.join(data_dir, "eval_overall_stats.json"), mode="w") as outfile:
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
                eval_overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / 1_000_000_000
                eval_overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / 1_000_000_000

    with open(os.path.join(data_dir, "eval_overall_means.json"), mode="w") as outfile:
        json.dump(eval_overall_means, outfile, indent=4)


def plot_overall_step_time_curves(data_dir):
    
    with open(os.path.join(data_dir, "overall_means.json"), mode="r") as infile:
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
def plot_epoch_individual_time_curves(data_dir, sharey=True):
    
    with open(os.path.join(data_dir, "epoch_overall_means.json"), mode="r") as infile:
        overall_means = json.load(infile)

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(durations.keys()), layout="constrained", figsize=(3.1 * len(durations.keys()), 8), sharey=sharey)
    fig.suptitle("Average Time per Step Phase (epochs > 1) ")

    i_ax = -1
    for metric in durations.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric])

        # plot the metric in the axes
        for gpu_key in GPUs_str:
            x = np.asarray(batches_int)

            y = [ overall_means[gpu_key][batch][metric]["mean"] for batch in batches_str ]
            y = np.asarray(y)

            std = [ overall_means[gpu_key][batch][metric]["stdev"] for batch in batches_str]
            std = np.asarray(std)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")
            ax.fill_between(x, y-std, y+std, alpha=0.15)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            ax.legend()

    # Create output directory if it doesn't exist
    pathlib.Path(os.path.join(data_dir, "plots")).mkdir(parents=True, exist_ok=True)

    filename = "epoch_indiv_times_sharey.png" if sharey else "epoch_indiv_times.png"
    figure_filename = os.path.join(data_dir, "plots", filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def plot_eval_individual_time_curves(data_dir, sharey=True):
    
    with open(os.path.join(data_dir, "eval_overall_means.json"), mode="r") as infile:
        eval_overall_means = json.load(infile)

    # Overall plot
    num_subplots = len(eval_durations.keys()) - 1
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, layout="constrained", figsize=(3.1 * num_subplots, 8), sharey=sharey)
    fig.suptitle("Average Time per Evaluation Step Phase (Eval > 1) ")

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

            y = [ eval_overall_means[gpu_key][batch][metric]["mean"] for batch in batches_str ]
            y = np.asarray(y)

            std = [ eval_overall_means[gpu_key][batch][metric]["stdev"] for batch in batches_str]
            std = np.asarray(std)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")
            ax.fill_between(x, y-std, y+std, alpha=0.15)

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            ax.legend()

    # Create output directory if it doesn't exist
    pathlib.Path(os.path.join(data_dir, "plots")).mkdir(parents=True, exist_ok=True)

    filename = "eval_indiv_times_sharey.png" if sharey else "eval_indiv_times.png"
    figure_filename = os.path.join(data_dir, "plots", filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def plot_eval_individual_time_curves_by_image_size(data_dir, sharey=True):
    
    with open(os.path.join(data_dir, "eval_overall_stats.json"), mode="r") as infile:
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

    filename = "eval_indiv_times_img_size_sharey.png" if sharey else "eval_indiv_times_img_size.png"
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
    args = parser.parse_args()

    # export_per_eval_stats(args.data_dir)
    # export_per_epoch_stats(args.data_dir)
    export_overall_epoch_stats(args.data_dir)
    # export_overall_eval_stats(args.data_dir)

    # plot_histograms(args.data_dir)
    # plot_overall_step_time_curves(args.data_dir)
    # plot_individual_time_curves(args.data_dir, sharey=False)
    # plot_epoch_individual_time_curves(args.data_dir, sharey=True)
    # plot_eval_individual_time_curves(args.data_dir, sharey=True)
    # plot_eval_individual_time_curves_by_image_size(args.data_dir, sharey=True)
