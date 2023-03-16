
import copy
import re
import statistics
import json
import argparse
import numpy as np
from pathlib import Path
from pprint import pprint
from matplotlib import pyplot as plt


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

def get_profile_subdirectories(trace_dir):
    trace_dir = trace_dir / "data"
    subdirs = []
    for item in trace_dir.iterdir():
        if item.is_dir():
            subdirs.append(item)

    return subdirs


def get_profiler_traces(trace_dir):

    traces = list(trace_dir.rglob('*.json'))
    assert len(traces) > 0, f"No profiler traces found in {trace_dir}"
    return traces


def get_breakdown_from_profiler_trace(trace_dir, output_dir, process=True):

    plotting_data_file = trace_dir / "processed"/ "BERT_plotting_data.json"

    if process or not plotting_data_file.is_file():

        all_data = {}

        profiler_dirs = get_profile_subdirectories(trace_dir)

        p_memcpy = re.compile(r'.*IteratorGetNext@@MemcpyHtoD')

        num_gpus = []
        batch_sizes = []

        import os
        outfile = open(os.path.join(data_dir, "bert_step_analysis.txt"), "w")
        outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Median':>15}\t{'Std':>15}\t{'1st quartile':>15}\t{'3rd quart':>15}\n")

        for profiler_dir in profiler_dirs:

            # extract num gpu and batch size
            num_gpu = get_num_gpus(profiler_dir.name)
            batch_size = get_batch_size(profiler_dir.name)
            # global_batch_size = num_gpu * batch_size
            global_batch_size = batch_size

            num_gpus.append(num_gpu)
            batch_sizes.append(batch_size)

            if num_gpu not in all_data:
                # init data structure
                all_data[num_gpu] = {
                    batch_size: {
                        "all_compute": [],
                        "load_batch_to_mem": [],
                        "step_end": [],
                        "data_loading_throughput": [],
                        "data_proc_throughput": [],
                    }
                }
            else:
                all_data[num_gpu][batch_size] = {
                    "all_compute": [],
                    "load_batch_to_mem": [],
                    "step_end": [],
                    "data_loading_throughput": [],
                    "data_proc_throughput": [],
                }

            # iterate over all profiler traces
            all_traces = get_profiler_traces(profiler_dir)

            for trace in all_traces:

                print(trace)
                with open(trace, 'r') as trace:
                    trace = json.load(trace)

                all_events = trace['traceEvents']

                lowest_ts = float('inf')
                highest_ts = 0

                memcpy_ts = None
                for event in all_events:

                    if 'cat' in event and event['cat'] == 'Op':
                        # Keep track of timestamps
                        lowest_ts = min(lowest_ts, event['ts'])
                        highest_ts = max(highest_ts, event['ts'])

                        if event['name'] == 'IteratorV2':
                            # print(f'Found IteratorV2: {event}')
                            iterator_start = np.datetime64(event['ts'], 'us')
                            continue

                        if event['name'] == 'IteratorGetNext':
                            # print(f'Found IteratorGetNext: {event}')
                            iterator_end = np.datetime64(event['ts'], 'us') + event['dur']
                            continue

                        if event['name'] == 'unknown':
                            if 'args' in event and 'name' in event['args']:
                                if re.match(p_memcpy, event['args']['name']):
                                    # memcopy to device the batch
                                    memcpy_ts = np.datetime64(event['ts'], 'us')

                load_batch_to_mem = iterator_end - iterator_start
                load_batch_to_mem = load_batch_to_mem.astype('int64') / 1e6
                data_loading_throughput = global_batch_size / ( load_batch_to_mem )

                start_ts = np.datetime64(lowest_ts, 'us')
                end_ts = np.datetime64(highest_ts, 'us')
                step_end = end_ts - start_ts
                step_end = step_end.astype('int64') / 1e6

                data_proc_throughput = global_batch_size / step_end

                # We consider that since no computation can be done before the batch is loaded in memory
                # the total computation time is everything from sending the batch to the trace end
                all_compute = end_ts - memcpy_ts
                all_compute = all_compute.astype('int64') / 1e6

                all_data[num_gpu][batch_size]["data_loading_throughput"].append(data_loading_throughput)
                all_data[num_gpu][batch_size]["data_proc_throughput"].append(data_proc_throughput)
                all_data[num_gpu][batch_size]["load_batch_to_mem"].append(load_batch_to_mem)
                all_data[num_gpu][batch_size]["all_compute"].append(all_compute)
                all_data[num_gpu][batch_size]["step_end"].append(step_end)


        # DS to hold median, q1, q3 of data
        plotting_data = copy.deepcopy(all_data)

        simulation_sleep_time = {}


        # now we have the raw data
        for gpu_key in all_data:
            for batch_size in all_data[gpu_key]:

                outfile.write(f"BERT_{gpu_key}GPU_{batch_size}batch\n")

                for metric in all_data[gpu_key][batch_size]:

                    data = all_data[gpu_key][batch_size][metric]

                    mean = statistics.mean(data)
                    median = statistics.median(data)
                    std = statistics.stdev(data)
                    quartiles = statistics.quantiles(data)

                    plotting_data[gpu_key][batch_size][metric] = {
                        "median": median,
                        "mean": mean,
                        "std": std,
                        "q1": quartiles[0],
                        "q3": quartiles[2],
                    }
                    ROUND = 5
                    outfile.write(f"{metric:>30}:\t{round(mean, ROUND):>15}\t{round(median, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
                outfile.write("\n")

        outfile.flush()
        outfile.close()
        

        for gpu_key in all_data:
            for batch_size in all_data[gpu_key]:
                print(gpu_key, batch_size)
                if gpu_key in simulation_sleep_time:
                    simulation_sleep_time[gpu_key][batch_size] = {
                        'mean': plotting_data[gpu_key][batch_size]['all_compute']['mean'],
                        'std': plotting_data[gpu_key][batch_size]['all_compute']['std'],
                        'median': plotting_data[gpu_key][batch_size]['all_compute']['median'],
                        'q1': plotting_data[gpu_key][batch_size]['all_compute']['q1'],
                        'q3': plotting_data[gpu_key][batch_size]['all_compute']['q3'],
                    }
                else:
                    simulation_sleep_time[gpu_key] = {
                        batch_size: {
                            'mean': plotting_data[gpu_key][batch_size]['all_compute']['mean'],
                            'std': plotting_data[gpu_key][batch_size]['all_compute']['std'],
                            'median': plotting_data[gpu_key][batch_size]['all_compute']['median'],
                            'q1': plotting_data[gpu_key][batch_size]['all_compute']['q1'],
                            'q3': plotting_data[gpu_key][batch_size]['all_compute']['q3'],
                        }
                    }     

        plotting_data_dir = trace_dir / "processed"
        plotting_data_dir.mkdir(exist_ok = True)

        with open(plotting_data_dir / "BERT_plotting_data.json", 'w') as outfile:
            # Save plotting data for faster plotting next time
            json.dump(plotting_data, outfile)

        import os
        with open(os.path.join(plotting_data_dir, "dlio_sleep_times.json"), "w") as outfile:
            json.dump(simulation_sleep_time, outfile, indent=4)




def plot_throughputs(trace_dir):
    # Load the data
    with open(trace_dir / "processed" / "BERT_plotting_data.json", 'r') as infile:
        plotting_data = json.load(infile)


    pprint(plotting_data)

    metrics_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Processing Throughput",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }
    
    batch_sizes = set()
    num_gpus = set()

    for gpu in plotting_data:
        num_gpus.add(gpu)
        for batch in plotting_data[gpu]:
            batch_sizes.add(batch)

    num_gpus = list(num_gpus)
    batch_sizes = list(batch_sizes)

    num_gpus_str = sorted([str(g) for g in num_gpus])
    batch_sizes_str = sorted([str(b) for b in batch_sizes])

    # Overall plot
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(5 * len(metrics_to_plot.keys()), 6))

    FONTSIZE = 18
    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        # plot the metric in the axes
        for gpu_key in num_gpus_str:

            x = np.asarray(batch_sizes)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batch_sizes_str ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batch_sizes_str ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batch_sizes_str ]
            q3 = np.asarray(q3)

            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            ax.fill_between(x, q1, q3, alpha=0.1)

            if len(batch_sizes_str[0]) > 3:
                ax.set_xticks(batch_sizes, batch_sizes_str, rotation=-46, ha='center', fontsize=FONTSIZE)
            else:
                ax.set_xticks(batch_sizes, batch_sizes_str, ha='center', fontsize=FONTSIZE-1)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.2, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supylabel("Samples / s", fontsize=FONTSIZE)
    fig.supxlabel('Batch size', fontsize=FONTSIZE)

    output_dir = Path(data_dir) / "plots"
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"BERT_throughputs.png"
    figure_filename = output_dir / filename

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



def plot_bw_and_breakdown(trace_dir):
    # Load the data
    with open(trace_dir / "processed" / "BERT_plotting_data.json", 'r') as infile:
        plotting_data = json.load(infile)

    pprint(plotting_data)

    metrics_pretty_names = {
        "data_loading_throughput": "Data Load Bandwidth",
        "data_proc_throughput": "Processing Bandwidth",
        "step_end": "Overall Step Time",
        "load_batch_to_mem": "Data Loading",
        "all_compute": "Computation",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }
    
    batch_sizes = set()
    num_gpus = set()

    for gpu in plotting_data:
        num_gpus.add(gpu)
        for batch in plotting_data[gpu]:
            batch_sizes.add(batch)

    num_gpus = list(num_gpus)
    batch_sizes = list(batch_sizes)

    num_gpus_str = sorted([str(g) for g in num_gpus])
    batch_sizes_str = sorted([str(b) for b in batch_sizes])

    # Overall plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 6))

    for i in range(3, len(axes)):
        axes[i].sharey(axes[2])

    FONTSIZE = 18
    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        if i_ax == 0:
            ax.set_ylabel("Bandwidth (samples/s)", fontsize=FONTSIZE)

        if i_ax == 2:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylabel("Time (s)", fontsize=FONTSIZE)

        if i_ax >= 3:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left = False)

        # plot the metric in the axes
        for gpu_key in num_gpus_str:

            x = np.asarray(batch_sizes)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batch_sizes_str ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batch_sizes_str ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batch_sizes_str ]
            q3 = np.asarray(q3)

            
            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            ax.fill_between(x, q1, q3, alpha=0.1)
            # ax.set_xscale('log', base=2)

            if len(batch_sizes_str[0]) > 3:
                ax.set_xticks(batch_sizes, batch_sizes_str, rotation=-46, ha='center', fontsize=FONTSIZE)
                # ax.tick_params(axis='x', labelrotation=45)
            else:
                ax.set_xticks(batch_sizes, batch_sizes_str, ha='center', fontsize=FONTSIZE-1)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.2, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = Path(data_dir) / "plots"
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"BERT_breakdown_paper_and_BW.png"
    figure_filename = output_dir / filename

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_breakdown(trace_dir, sharey=True):
    # Load the data
    with open(trace_dir / "processed" / "BERT_plotting_data.json", 'r') as infile:
        plotting_data = json.load(infile)

    pprint(plotting_data)

    metrics_pretty_names = {
        "step_end": "Overall Step",
        "load_batch_to_mem": "1 Batch Loading",
        "all_compute": "2 Computation",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }
    
    batch_sizes = set()
    num_gpus = set()

    for gpu in plotting_data:
        num_gpus.add(gpu)
        for batch in plotting_data[gpu]:
            batch_sizes.add(batch)

    num_gpus = list(num_gpus)
    batch_sizes = list(batch_sizes)

    num_gpus_str = sorted([str(g) for g in num_gpus])
    batch_sizes_str = sorted([str(b) for b in batch_sizes])

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot.keys()), 6), 
        sharey=sharey
    )

    FONTSIZE = 18
    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        if sharey:
            if i_ax == 1:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if i_ax >= 2:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(left = False)

        # plot the metric in the axes
        for gpu_key in num_gpus_str:

            x = np.asarray(batch_sizes)

            y = [ plotting_data[gpu_key][batch][metric]["median"] for batch in batch_sizes_str ]
            y = np.asarray(y)

            q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batch_sizes_str ]
            q1 = np.asarray(q1)

            q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batch_sizes_str ]
            q3 = np.asarray(q3)

            
            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            ax.fill_between(x, q1, q3, alpha=0.1)
            # ax.set_xscale('log', base=2)

            if len(batch_sizes_str[0]) > 3:
                ax.set_xticks(batch_sizes, batch_sizes_str, rotation=-46, ha='center', fontsize=FONTSIZE)
                # ax.tick_params(axis='x', labelrotation=45)
            else:
                ax.set_xticks(batch_sizes, batch_sizes_str, ha='center', fontsize=FONTSIZE-1)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.2, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = Path(data_dir) / "plots"
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"BERT_step_breakdown{'_sharey' if sharey else ''}.png"
    figure_filename = output_dir / filename

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot step breakdown from BERT Profiler traces")
    parser.add_argument("data_dir", help="Data directory")
    parser.add_argument("--do-processing", "-dp", action='store_true', help="Whether to proces raw data or use saved file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() and data_dir.is_dir():
        print(f"Invalid data directory given: {data_dir}")

    output_dir = Path('bert_breakdown/')
    output_dir.mkdir(exist_ok=True)
    
    print(args.do_processing)
    get_breakdown_from_profiler_trace(data_dir, output_dir, process=args.do_processing)

    # plot_bw_and_breakdown(data_dir)
    plot_breakdown(data_dir)
    plot_breakdown(data_dir, sharey=False)
    plot_throughputs(data_dir)