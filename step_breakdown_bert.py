
from time import time
import os
import re
import json
import copy
import argparse
import statistics
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from pathlib import Path
from pprint import pprint
from matplotlib import pyplot as plt

from step_breakdown import add_headers, fit_linear_to_distribs, fit_normal_distrib_and_plot, single_histogram
from step_breakdown_dlio import OOMFormatter


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

def get_profile_subdirectories(trace_dirs):

    subdirs = []
    for trace_dir in trace_dirs:
        trace_dir = trace_dir / "data"
        for item in trace_dir.iterdir():
            if item.is_dir():
                subdirs.append(item)

    return sorted(subdirs)


def get_profiler_traces(trace_dir):

    traces = list(trace_dir.rglob('*timeline-*.json'))
    assert len(traces) > 0, f"No profiler traces found in {trace_dir}"
    return traces


def get_breakdown_from_profiler_trace(trace_dirs, output_dir, process=True, big_histo=False):

    plotting_data_file = output_dir / "processed"/ "BERT_plotting_data.json"

    if process or not plotting_data_file.is_file():

        all_data = {}
        fit_data = {}

        profiler_dirs = get_profile_subdirectories(trace_dirs)

        p_memcpy = re.compile(r'.*IteratorGetNext@@MemcpyHtoD')

        num_gpus = []
        batch_sizes = []

        import os
        outfile = open(os.path.join(output_dir, "bert_step_analysis.txt"), "w")
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
                        "load_batch_mem": [],
                        "step_end": [],
                        "data_loading_throughput": [],
                        "data_proc_throughput": [],
                    }
                }
                fit_data[num_gpu] = {
                    batch_size: {
                        'mu': [],
                        'std': [],
                    }
                }
            elif batch_size not in all_data[num_gpu]:
                all_data[num_gpu][batch_size] = {
                    "all_compute": [],
                    "load_batch_mem": [],
                    "step_end": [],
                    "data_loading_throughput": [],
                    "data_proc_throughput": [],
                }
                fit_data[num_gpu][batch_size] = {
                    'mu': [],
                    'std': [],
                }

            # iterate over all profiler traces
            # each is a single step, there should be 30 for each batch_size * num_gpu configuration
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

                load_batch_mem = iterator_end - iterator_start
                load_batch_mem = load_batch_mem.astype('int64') / 1e6
                data_loading_throughput = global_batch_size / ( load_batch_mem )

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
                all_data[num_gpu][batch_size]["load_batch_mem"].append(load_batch_mem)
                all_data[num_gpu][batch_size]["all_compute"].append(all_compute)
                all_data[num_gpu][batch_size]["step_end"].append(step_end)


            # Once we've gone through the whole log file, plot a histogram of the computation time
            # and fit a gaussian to it (visual inspection reveals it looks normally distributed)
            mu, std = fit_normal_distrib_and_plot(all_data[num_gpu][batch_size]['all_compute'], output_dir, title=f"all_compute_{num_gpu}g_{batch_size}b")
            fit_data[num_gpu][batch_size]['mu'].append(mu)
            fit_data[num_gpu][batch_size]['std'].append(std)


        # Save the fit data
        with open(output_dir / "compute_time_fit.json", "w") as fitfile:
            json.dump(fit_data, fitfile, indent=4)

        filename = Path(output_dir) / 'ditribution_fits.txt'
        with open(filename, 'w') as fitfile:
            fit_linear_to_distribs(fit_data, fitfile)

        # Save all data
        with open(output_dir / "all_data.json", "w") as alldatafile:
            json.dump(all_data, alldatafile, indent=4)

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
                # print(gpu_key, batch_size)
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

        plotting_data_dir = output_dir / "processed"
        plotting_data_dir.mkdir(exist_ok = True)

        with open(plotting_data_dir / "BERT_plotting_data.json", 'w') as outfile:
            # Save plotting data for faster plotting next time
            json.dump(plotting_data, outfile)

        with open(os.path.join(plotting_data_dir, "dlio_sleep_times.json"), "w") as outfile:
            json.dump(simulation_sleep_time, outfile, indent=4)

    if big_histo:

        with open(output_dir / "all_data.json", 'r') as infile:
            all_data = json.load(infile)

        all_batches = set()
        all_gpus = set()

        for gpu in all_data:
            all_gpus.add(gpu)
            for batch in all_data[gpu]:
                all_batches.add(batch)

        # Plot a super histogram with all the fits
        all_batches = sorted(list(all_batches), key=int)
        all_gpus = sorted(list(all_gpus), key=int)

        fig, axes = plt.subplots(ncols=len(all_gpus), nrows=len(all_batches), figsize=(2 + 2*len(all_gpus), 1 + 2 * len(all_batches)))

        row_headers = [f'Batch size: {b}' for b in all_batches]
        col_headers = [f'{g} GPU{"s" if int(g) > 1 else ""}' for g in all_gpus]

        font_kwargs = dict(fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        fig.tight_layout()

        row = -1
        for batch in all_batches:
            row += 1
            col = -1
            for gpu in all_gpus:
                col += 1
                ax = axes[row, col]
                # Can have missing data for a certain combination
                if batch not in all_data[gpu] or 'all_compute' not in all_data[gpu][batch]:
                    continue
                single_histogram(ax, all_data[gpu][batch]['all_compute'])

        # Create output directory if it doesn't exist
        outdir = Path(output_dir) / "histograms"
        outdir.mkdir(parents=True, exist_ok=True)

        figure_filename = outdir / f'BERT_all_histograms.png'

        plt.savefig(figure_filename, format="png", dpi=500)
        print(f'Saved {figure_filename}')
        # Clear the current axes.
        plt.cla() 
        # Closes all the figure windows.
        plt.close('all')   
        plt.close(fig)



def plot_throughputs(data_dir, legend=False, title=None):
    # Load the data
    with open(data_dir / "processed" / "BERT_plotting_data.json", 'r') as infile:
        plotting_data = json.load(infile)

    metrics_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Compute Throughput",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }
    
    batch_sizes = set()
    num_gpus = set()

    for gpu in plotting_data:
        num_gpus.add(gpu)
        for batch in plotting_data[gpu]:
            batch_sizes.add(batch)

    num_gpus = list(num_gpus)
    batch_sizes = sorted(list(batch_sizes))

    num_gpus_str = sorted([str(g) for g in num_gpus])
    batch_sizes_str = sorted([str(b) for b in batch_sizes])

    # Overall plot
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(max(6 * len(metrics_to_plot), 10), 4),
    )

    FONTSIZE = 24

    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]

        if title not in ['gen']:
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

            ax.set_xticks(batch_sizes, batch_sizes_str, ha='center', fontsize=FONTSIZE-2)

            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-2)

            ax.set_xlim((batch_sizes[0], batch_sizes[-1]))

            if metric == 'data_loading_throughput':

                if title in ['gen']:
                    ticks = [10, 30, 50]
                    tick_labels = [str(t) for t in ticks]
                    ticks = [t * 1e3 for t in ticks]
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(tick_labels)

                ax.yaxis.set_major_formatter(OOMFormatter(3, "%.0f"))
                ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

            else:
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.22, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supylabel("Samples / s", fontsize=FONTSIZE)
    # fig.supxlabel('Batch size', fontsize=FONTSIZE)

    output_dir = Path(data_dir) / "plots"
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"BERT_{title + '_' if title else ''}throughputs.png"
    figure_filename = output_dir / filename

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)




def plot_breakdown(data_dir, sharey=True, legend=False, title=None):
    # Load the data
    with open(data_dir / "processed" / "BERT_plotting_data.json", 'r') as infile:
        plotting_data = json.load(infile)

    metrics_pretty_names = {
        "step_end": "Overall Step",
        "load_batch_mem": "Loading",
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
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot.keys()), 4), 
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

            if metric == 'load_batch_mem':
                ax.yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
                ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)


            ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
            ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)


    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.2, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE)

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = Path(data_dir) / "plots"
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"BERT_step_breakdown{'_sharey' if sharey else ''}.png"
    figure_filename = output_dir / filename

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



def plot_data_loading(data_dir, mean_or_median="median", fill_between=True, legend=False):
    print(f'Plotting Step breakdown summary')

    with open(data_dir / "processed" / "BERT_plotting_data.json", 'r') as infile:
        plotting_data = json.load(infile)

    ## Shorter plot 
    metrics_pretty_names = {
        "load_batch_mem": "Batch Loading",
    }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, ax = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(10, 6),
    )

    GPUs_to_plot = plotting_data.keys()
    plotted_batch_sizes = set()

    FONTSIZE = 18

    for metric in metrics_to_plot:
        ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:

            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
            plotted_batch_sizes.update(batches_to_plot)

            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric][mean_or_median] for batch in batches_to_plot ]
            y = np.asarray(y)

            ax.plot(x, y, label=f"{gpu_key} GPUs", )

            if fill_between:

                if mean_or_median == 'median':
                    q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
                    q1 = np.asarray(q1)

                    q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
                    q3 = np.asarray(q3)

                    ax.fill_between(x, q1, q3, alpha=0.1)
                else:
                    std = [ plotting_data[gpu_key][batch][metric]["std"] for batch in batches_to_plot ]
                    std = np.asarray(std)

                    ax.fill_between(x, y-std, y+std, alpha=0.1)

        # We support plotting different number of points for
        # different numbers of GPUs, so we keep track of the
        # longest amount of points to properly label the x axis
        x_axis_labels = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in x_axis_labels]

        ax.set_xticks(x_axis_labels, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)
        plotted_batch_sizes = set()

    handles, labels = ax.get_legend_handles_labels()
    if legend:
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

    output_dir = os.path.join(data_dir, "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"BERT_data_loading_{mean_or_median}{'_nofill' if fill_between is False else ''}.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)





if __name__ == "__main__":

    parser = argparse.ArgumentParser("Plot step breakdown from BERT Profiler traces")
    parser.add_argument("data_dirs", nargs='+', help="Data directories")
    parser.add_argument("-o", "--output-dir", default=None ,help="(optional) Output directory.")
    parser.add_argument("-t", "--title", default=None, help="(optional) Plot title.")
    parser.add_argument("-b", "--breakdown", action="store_true", help="Plot the step breakdown.")
    parser.add_argument("-tp", "--throughputs", action="store_true", help="Plot the throughputs.")
    parser.add_argument("-bh", "--big-histo", action="store_true", help="Save file with all compute time distributions and fits for the annex.")
    parser.add_argument("-dp", "--do-processing", action='store_true', default=False, help="Whether to proces raw data (long) or use saved file")
    args = parser.parse_args()

    if not (args.breakdown or args.throughputs):
        print('No type of plot requested. Exiting.')
        exit()

    title = args.title

    data_dirs = []
    for data_dir in args.data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists() and data_dir.is_dir():
            print(f"Invalid data directory given: {data_dir}")
        data_dirs.append(data_dir)
    
    if len(data_dirs) == 0:
        print(f'ERROR: No valid data directories given')
        exit(-1)

    if args.output_dir is None:
        args.output_dir = "bert_breakdown/"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    get_breakdown_from_profiler_trace(data_dirs, output_dir, process=args.do_processing, big_histo=args.big_histo)

    if args.breakdown:
        plot_breakdown(output_dir, sharey=False)
        
    if args.throughputs:
        plot_throughputs(output_dir, title=title)

