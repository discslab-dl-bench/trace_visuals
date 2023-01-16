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

# For UNET3D
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

eval_metrics_pretty_names = {
    "eval_step_end": "Overall eval step",
    "eval_load_batch_mem": "1-Batch loaded in memory",
    "eval_load_batch_gpu": "2-Batch loaded to GPU",
    "eval_sliding_window": "3-Sliding window calc",
    "eval_loss_and_score_fn": "4-Loss and score calc",
    "eval_image_sizes": "Image Sizes"
}
durations = { metric: [] for metric in metrics_pretty_names }
eval_durations = { metric: [] for metric in eval_metrics_pretty_names }

WORKLOAD = "UNET"
GPUs_int = [2,4,6,8]
batches_int = [1, 2, 3, 4, 5]
GPUs_str = [str(g) for g in GPUs_int]
batches_str = [ str(b) for b in batches_int ]
PATTERN=f'.*_instrumented\.json'
NUM_EPOCHS = 10
NUM_EVALS = 3



# # For DLRM
# metrics_pretty_names = {
#     "step_end": "Overall step",
#     "load_batch_mem": "1-Batch loaded in memory",
#     "model_forward_pass": "3-Forward pass",
#     "loss_tensor_calc": "4-Loss calculation",
#     "model_backward_pass": "5-Backward pass",
#     "model_optim_step": "6-Optimizer step",
# }
# eval_metrics_pretty_names = {
#     "eval_step_end": "Overall eval step",
#     "eval_load_batch_mem": "1-Batch loaded in memory",
#     "eval_forward_pass": "3-Forward Pass",
#     "eval_all_gather": "4-All gather",
#     "eval_score_compute": "5-Score Computation",
# }
# durations = { metric: [] for metric in metrics_pretty_names }
# eval_durations = { metric: [] for metric in eval_metrics_pretty_names }

# GPUs_int = [2, 4, 6, 8]
# batches_int = [2048, 4096, 8192, 16384, 32768, 65536, 262144]
# GPUs_str = [str(g) for g in GPUs_int]
# batches_str = [ str(b) for b in batches_int ]

# WORKLOAD = "DLRM"
# PATTERN=f'DLRM_.*\.json'
# NUM_EPOCHS = 1
# NUM_EVALS = 2

def export_per_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data')) if re.match(PATTERN, f)]

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
        # Gets a prettier name from the log file name
        outfile_base = os.path.basename(log_file).replace("_instrumented", "").replace(".json", "")

        # Create output directory if it doesn't exist
        pathlib.Path(os.path.join(data_dir, "per_epoch")).mkdir(parents=True, exist_ok=True)

        outfile_stats = outfile_base + "_per_epoch_stats.json"
        outfile_durations = outfile_base + "_per_epoch_durations.json"

        json.dump(per_epoch_stats, open(os.path.join(data_dir, "per_epoch", outfile_stats), mode="w"), indent=4)
        json.dump(per_epoch_durations, open(os.path.join(data_dir, "per_epoch", outfile_durations), mode="w"), indent=4)
        

def export_per_eval_stats(data_dir):
    print("Exporting per eval stats")
    log_files = [os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data')) if re.match(PATTERN, f)]

    print(log_files)

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
                # # Save the size only once, else we will have the same value 4 times
                # if line['key'] == 'eval_load_batch_mem':
                #     # The shape gives the number of pixels, each is of type np.float32 so 4 bytes
                #     sample_size = int(np.prod(line['value']['image_shape'])) * 4
                #     per_eval_durations[eval_num]['eval_image_sizes'].append(sample_size)

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




def get_smallest_common_length(data):
    l = float("inf")
    for k,v in data.items():
        if len(v) != l:
            l = min(l, len(v))
    return l


def UNET_export_overall_epoch_stats(data_dir):

    log_files = [os.path.join(data_dir, "per_epoch", f) for f in os.listdir(os.path.join(data_dir, "per_epoch")) if re.match(rf'{WORKLOAD}_.*_durations\.json', f)]

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

    overall_durations["total_minus_cum_loss"] = np.asarray([], dtype=np.int64)
    overall_durations["total_minus_load_preproc_and_cum_loss"] = np.asarray([], dtype=np.int64)


    for log_file in log_files:

        print(f"Processing {log_file}")
        infile = open(log_file, mode='r')
        log = json.load(infile)
        infile.close()
 
        res = re.search(r'.*([0-9])GPU.*', log_file)
        gpu_key = int(res.group(1))
        batch_key = int(os.path.basename(log_file).split("_")[2].replace("batch", ""))
        print(f"GPUs: {gpu_key}, batch size: {batch_key}")
        
        total_minus_cum_loss = np.asarray([], dtype=np.int64)
        total_minus_load_preproc_and_cum_loss = np.asarray([], dtype=np.int64)

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

            # Remove preproc again
            # should have total - load batch + (preproc - preproc) - cum_loss 
            diff -= sample_preproc
            overall_durations["total_minus_load_preproc_and_cum_loss"] = np.append(overall_durations["total_minus_load_preproc_and_cum_loss"], diff)
            total_minus_load_preproc_and_cum_loss = np.append(total_minus_load_preproc_and_cum_loss, diff)

            # should have total - load batch + load batch + (preproc - preproc) - cum_loss 
            # = total - cum_loss
            diff += np.asarray(data["load_batch_mem"][:min_size])
            overall_durations["total_minus_cum_loss"] = np.append(overall_durations["total_minus_cum_loss"], diff)
            total_minus_cum_loss = np.append(total_minus_cum_loss, diff)

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

        overall_stats[gpu_key][batch_key]["total_minus_load_preproc_and_cum_loss"] = total_minus_load_preproc_and_cum_loss.tolist()
        overall_stats[gpu_key][batch_key]["total_minus_cum_loss"] = total_minus_cum_loss.tolist()

    print("computing actual overall means")

    actual_overall_stats = copy.deepcopy(durations)

    for metric in overall_durations.keys():
        print(metric)
        quartiles = stats.quantiles(overall_durations[metric])
        actual_overall_stats[metric] = {
            "mean": round(np.asarray(overall_durations[metric]).mean() / 1_000_000_000, 3),
            "stdev": round(np.asarray(overall_durations[metric]).std() / 1_000_000_000, 3),
            "q1": round(quartiles[0] / 1_000_000_000, 3),
            "q3": round(quartiles[2] / 1_000_000_000, 3),
        }

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_actual_overall.json"), mode="w") as outfile:
        json.dump(actual_overall_stats, outfile, indent=4)

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_stats.json"), mode="w") as outfile:
        json.dump(overall_stats, outfile, indent=4)
    
    all_exported_metrics = list(durations.keys())
    all_exported_metrics.extend(["total_minus_load_preproc_and_cum_loss", "total_minus_cum_loss"])

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
                overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / 1_000_000_000
                quartiles = stats.quantiles(data)
                overall_means[gpu_key][batch_key][metric]["q1"] = quartiles[0] / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["q3"] = quartiles[2] / 1_000_000_000

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_means.json"), mode="w") as outfile:
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
 
        res = re.search(r'.*([0-9])GPU.*', log_file)
        gpu_key = int(res.group(1))
        batch_key = int(os.path.basename(log_file).split("_")[2].replace("batch", ""))
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

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_actual_overall.json"), mode="w") as outfile:
        json.dump(actual_overall_stats, outfile, indent=4)

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_stats.json"), mode="w") as outfile:
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
                overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / 1_000_000_000

                quartiles = stats.quantiles(data)
                overall_means[gpu_key][batch_key][metric]["q1"] = quartiles[0] / 1_000_000_000
                overall_means[gpu_key][batch_key][metric]["q3"] = quartiles[2] / 1_000_000_000

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_means.json"), mode="w") as outfile:
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
 
        res = re.search(r'.*([0-9])GPU.*', log_file)
        gpu_key = int(res.group(1))

        batch_key = int(os.path.basename(log_file).split("_")[2].replace("batch", ""))
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
                eval_overall_means[gpu_key][batch_key][metric]["mean"] = stats.mean(data) / 1_000_000_000
                eval_overall_means[gpu_key][batch_key][metric]["stdev"] = stats.pstdev(data) / 1_000_000_000

                quartiles = stats.quantiles(data)
                eval_overall_means[gpu_key][batch_key][metric]["q1"] = quartiles[0] / 1_000_000_000
                eval_overall_means[gpu_key][batch_key][metric]["q3"] = quartiles[2] / 1_000_000_000


    with open(os.path.join(data_dir, f"{WORKLOAD}_eval_overall_means.json"), mode="w") as outfile:
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
def UNET_plot_epoch_individual_time_curves(data_dir, sharey=True):
    print("Plotting relative time curves for epochs")

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_means.json"), mode="r") as infile:
        overall_means = json.load(infile)

    metrics_pretty_names = {
        "total_minus_cum_loss": "Overall step",
        "sample_load": "1.1-Batch load",
        "sample_preproc": "1.2-Sample Preproc (CPU)",
        "total_minus_load_preproc_and_cum_loss": "2-Batch Processing",
        "load_batch_gpu": "2.1-Batch to GPU",
        "model_forward_pass": "2.2-Forward pass",
        "loss_tensor_calc": "2.3-Loss calc",
        "model_backward_pass": "2.4-Backward pass",
        "model_optim_step": "2.5-Optimizer step",
        # "cum_loss_fn_calc": "7-Cumulative loss",
    }
    durations = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(durations.keys()), layout="constrained", figsize=(3.1 * len(durations.keys()), 8), sharey=sharey)
    fig.suptitle(f"{WORKLOAD} Average Time per Step Component (no step 7)")

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

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "relative_times")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}_epoch_relative_times.png"
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

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_means.json"), mode="r") as infile:
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

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "relative_times")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}_epoch_relative_times.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


# Individual phases plots
def plot_epoch_violin(data_dir, sharey=True):
    
    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_stats.json"), mode="r") as infile:
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
        "load_batch_mem": "1-Batch loaded in memory",
        "sum_all_but_load": "All batch processing (add)",
        "total_minus_load": "All batch processing (sub)",
    }
    durations = { metric: [] for metric in metrics_pretty_names }

    output_dir = os.path.join(data_dir, f"plots/{WORKLOAD}/histograms")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(data_dir, f"{WORKLOAD}_epoch_overall_stats.json"), mode="r") as infile:
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

            filename = f"{WORKLOAD}_batch{batch}_epoch_{metric}_latencies.png"
            figure_filename = os.path.join(output_dir, filename)

            plt.savefig(figure_filename, format="png", dpi=450)
            # Clear the current axes.
            plt.cla() 
            # Closes all the figure windows.
            plt.close('all')   
            plt.close(fig)



# Individual phases plots
def plot_eval_individual_time_curves(data_dir, sharey=True):
    print("Plotting relative times curves for evals")
    with open(os.path.join(data_dir, f"{WORKLOAD}_eval_overall_means.json"), mode="r") as infile:
        eval_overall_means = json.load(infile)

    # Overall plot

    # UNET3D we skip a key
    # num_subplots = len(eval_durations.keys()) - 1
    
    # DLRM plot them all
    num_subplots = len(eval_durations.keys()) 

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

            q1 = [ eval_overall_means[gpu_key][batch][metric]["q1"] for batch in batches_str]
            q1 = np.asarray(q1)

            q3 = [ eval_overall_means[gpu_key][batch][metric]["q3"] for batch in batches_str]
            q3 = np.asarray(q3)
            
            ax.plot(x, y, label=f"{gpu_key} GPUs")

            # ax.fill_between(x, y-std, y+std, alpha=0.15)
            ax.fill_between(x, y-q1, y+q3, alpha=0.15)

            # ax.set_xscale('log', base=2)
            # for large batch sizes
            # ax.set_xticks(batches_int, batches_str, rotation=45, ha='left')
            if len(batches_str[0]) > 3:
                ax.tick_params(axis='x', labelrotation=45)

            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Batch size", fontsize=10)
            ax.set_ylabel("Time taken (s)", fontsize=10)
            ax.legend()

    output_dir = os.path.join(data_dir, "plots", WORKLOAD, "relative_times")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"{WORKLOAD}_eval_relative_times.png"
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
    args = parser.parse_args()

    # export_per_eval_stats(args.data_dir)
    # export_per_epoch_stats(args.data_dir)
    
    UNET_export_overall_epoch_stats(args.data_dir)
    # DLRM_export_overall_epoch_stats(args.data_dir)
    # export_overall_eval_stats(args.data_dir)

    # plot_overall_step_time_curves(args.data_dir)
    # plot_latency_histograms(args.data_dir)

    UNET_plot_epoch_individual_time_curves(args.data_dir, sharey=True)

    # DLRM_plot_epoch_individual_time_curves(args.data_dir, sharey=True)
    # plot_eval_individual_time_curves(args.data_dir, sharey=True)

    # plot_eval_individual_time_curves_by_image_size(args.data_dir, sharey=True)
