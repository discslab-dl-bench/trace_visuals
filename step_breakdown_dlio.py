import copy
import os
import re
import time
import pathlib
import argparse
from matplotlib import patches
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator, ScalarFormatter, StrMethodFormatter
from pathlib import Path



PATTERN_INSTRU_LINE = {
    'UNET3D': re.compile(r'\[INFO\] (sample_load|sample_preproc|load_sample|load_batch_mem|load_batch_inner|all_compute|step_end|) (\d+)'),
    'DLRM': re.compile(r'\[INFO\] (batch_load|batch_preproc|load_batch_mem|load_batch_inner|all_compute|step_end|) (\d+)'),
    'BERT': re.compile(r'\[INFO\] (load_batch_mem|load_batch_inner|all_compute|step_end|) (\d+)'),
}

DLRM_BATCH_SIZE_STRINGS = {
    2048: "2k",
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
    65536: "64k",
    130712: "128k",
    131072: "128k",
    262144: "256k",
    524288: "512k",
    1048576: "1M",
    2097152: "2M",
    4194304: '4M',
    8388608: '8M',
    8388608: '8M',
}


# Global variable to create a new plotting directory
timestamp = int(time.time())
plotting_dir = f'plots_{timestamp}'


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def get_smallest_common_length(data):
    l = float("inf")
    for k,v in data.items():
        if len(v) != l:
            l = min(l, len(v))
    return l


def get_num_gpus(log_file_name):
    res = re.search(r'.*_([0-9]+)GPU.*', log_file_name)
    if res is None:
        res = re.search(r'.*([0-9]+)gpu.*', log_file_name)
    if res is None:
        res = re.search(r'.*([0-9]+)g.*', log_file_name)
    return int(res.group(1))


def get_batch_size(log_file_name):
    res = re.search(r'.*([0-9]+)batch.*', log_file_name)
    if res is None:
        res = re.search(r'.*_([0-9]+)b_.*', log_file_name)
    if res is None:
        res = re.search(r'.*_([0-9]+)_.*', log_file_name)
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

FONTSIZE = 24


def get_all_log_files(data_dirs, workload):
    log_files = []

    for data_dir in data_dirs:
        data_dir  = Path(data_dir)
        for trace_dir in data_dir.iterdir():
            log_files.extend(list(trace_dir.rglob(f'dlio.log')))
    
    return sorted(log_files)


def generate_plotting_data(data_dirs, output_dir, workload):

    output_dir = Path(output_dir)

    log_files = get_all_log_files(data_dirs, workload)
    # print(log_files)

    all_data = {}

    all_metrics = {
        "load_batch_mem": [],
        "load_batch_inner": [],
        "all_compute": [],
        "step_end": [],
        "data_proc_throughput": [],
        "data_loading_throughput": [],
        "inner_data_loading_throughput": [],
        "step_throughput": [],
    }

    if workload == 'UNET3D':
        all_metrics['sample_load'] = []
        all_metrics['load_sample'] = []
        all_metrics['sample_preproc'] = []
        all_metrics['sample_load_throughput'] = []

    elif workload == 'DLRM':
        all_metrics['batch_load'] = []
        all_metrics['batch_preproc'] = []
        all_metrics['from_disk_throughput'] = []


    for log_file in log_files:
        # outfile.write(f"{log_file}\n")

        experiment_name = log_file.parts[-4]
        # print(experiment_name)

        num_gpu = get_num_gpus(experiment_name)
        batch_size = get_batch_size(experiment_name)

        if num_gpu not in all_data:
            all_data[num_gpu] = {
                batch_size: { metric: [] for metric in all_metrics }
            }
        else:
            if batch_size not in all_data[num_gpu]: 
                all_data[num_gpu][batch_size] = { metric: [] for metric in all_metrics }
        
        epoch = 0
        seen_evts = set()
        with open(log_file, 'r') as log_file:

            for line in log_file:

                if m := re.match(PATTERN_START_EPOCH, line):
                    # print(line)
                    epoch = int(m.group(1))
                    seen_evts = set()

                if workload == 'unet3d':
                    if epoch == 1:
                        continue
                
                if m := re.match(PATTERN_INSTRU_LINE[workload], line):
                    event = m.group(1)
                    duration = int(m.group(2)) / 1_000_000_000   # convert to seconds

                    # Skip the first step by making sure we've seen the 3 events
                    # everytime we're in a new epoch
                    if len(seen_evts) < 3:
                        seen_evts.add(event)
                        continue

                    all_data[num_gpu][batch_size][event].append(duration)

                    if event == 'load_batch_mem':
                        # data_tp = batch_size / duration
                        # all_data[num_gpu][batch_size]['data_loading_throughput'].append(data_tp)
                        pass
                    elif event == 'load_batch_inner':
                        inner_data_tp = batch_size / duration
                        all_data[num_gpu][batch_size]['data_loading_throughput'].append(inner_data_tp)
                    elif event == 'all_compute':
                        proc_tp = batch_size / duration
                        all_data[num_gpu][batch_size]['data_proc_throughput'].append(proc_tp)
                    elif event in ['sample_load', 'load_sample']:
                        sample_tp = 1 / duration 
                        all_data[num_gpu][batch_size]['sample_load_throughput'].append(sample_tp)
                    elif event in ['batch_load', 'load_batch']:
                        batch_tp = batch_size / duration 
                        all_data[num_gpu][batch_size]['from_disk_throughput'].append(batch_tp)
                    elif event == 'step_end':
                        batch_tp = batch_size / duration 
                        all_data[num_gpu][batch_size]['step_throughput'].append(batch_tp)


    # Deepcopy all_data but we'll replace the arrays with dictionaries of summary stats
    plotting_data = copy.deepcopy(all_data)

    with open(output_dir / f"DLIO_{workload}_step_analysis.log", "w") as outfile:
        # Print header
        outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Std':>15}\t{'q1':>15}\t{'Median':>15}\t{'q3':>15}\n")

        for gpu_key in all_data:
            for batch_key in all_data[gpu_key]:
                
                outfile.write(f"{workload}_{gpu_key}GPU_{batch_key}batch\n")

                for metric in all_data[gpu_key][batch_key]:
                    # print(gpu_key, batch_key, metric)
                    if len(all_data[gpu_key][batch_key][metric]) == 0:
                        continue
                    mean = stats.mean(all_data[gpu_key][batch_key][metric])
                    std = stats.stdev(all_data[gpu_key][batch_key][metric])
                    quartiles = stats.quantiles(all_data[gpu_key][batch_key][metric])

                    plotting_data[gpu_key][batch_key][metric] = {
                        'mean': mean,
                        'std': std,
                        'q1': quartiles[0],
                        'median': quartiles[1],
                        'q3': quartiles[2],
                    }
                    ROUND = 5
                    outfile.write(f"{metric:>30}:\t{round(mean, ROUND):>15}\t{round(std, ROUND):>15}\t{round(quartiles[0], ROUND):>15}\t{round(quartiles[1], ROUND):>15}\t{round(quartiles[2], ROUND):>15}\n")
            outfile.write("\n")

    from  pprint import pprint
    # pprint(plotting_data)

    return plotting_data



def plot_throughputs(plotting_data, output_dir, workload, legend=False, title=None):

    GPUs_int = []
    batches_int = set()

    for num_gpu in plotting_data:
        GPUs_int.append(int(num_gpu))

        for batch_size in plotting_data[num_gpu]:
            batches_int.add(int(batch_size)) 

    GPUs_int.sort()
    batches_int = sorted(list(batches_int))
    batches_str = [str(b) for b in batches_int]

    print(f'Plotting for gpus {GPUs_int} and batches {batches_int}')

    print("Plotting step breakdown from raw data for epochs > 1")

    ###############################################################
    # Modify metrics, gpus or batch sizes to plot here
    ###############################################################

    if workload == 'UNET3D':
        metrics_to_plot_pretty_names = {
            "sample_load_throughput": "VFS Throughput",
            "data_loading_throughput": "Data Throughput",
            "data_proc_throughput": "Compute Throughput",
        }
    elif workload == 'DLRM':
        metrics_to_plot_pretty_names = {
            "from_disk_throughput": "VFS Throughput",
            "data_loading_throughput": "Data Throughput",
            "data_proc_throughput": "Compute Throughput",
        }
    else:
        metrics_to_plot_pretty_names = {
            "data_loading_throughput": "Data Throughput",
            "data_proc_throughput": "Compute Throughput",
        }

    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    GPUs_to_plot = sorted(GPUs_int)
    batches_to_plot = batches_int
    batches_to_plot_str = batches_str


    # Overall plot
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        # figsize=(max(6 * len(metrics_to_plot), 10), 6),
        figsize=(max(6 * len(metrics_to_plot), 10), 4),
    )

    plotted_batch_sizes = set()
    
    i_ax = -1
    for metric in metrics_to_plot.keys():
        i_ax += 1
        ax = axes[i_ax]
        # ax.set_title(metrics_to_plot_pretty_names[metric], fontsize=FONTSIZE)

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
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]

        # for large batch sizes
        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.offsetText.set_fontsize(FONTSIZE-3)

            if metric == 'from_disk_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(7, "%1.1f")) 

            if metric == 'data_loading_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))  
            
            if metric == 'data_proc_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(5, "%1.1f"))

        elif workload == 'UNET3D':
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

            if metric == 'data_loading_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(3, "%2.1f"))
                ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

            if metric == 'data_proc_throughput':
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

            if metric == 'data_loading_throughput':
                if title == 'vldb':
                    ticks = [8, 4, 12]
                    tick_labels = [str(t) for t in ticks]
                    ticks = [t * 1e3 for t in ticks]
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(tick_labels)
                    
                ax.yaxis.set_major_formatter(OOMFormatter(3, "%.0f"))
                ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

            if metric == 'data_proc_throughput':
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))


        ax.set_xlim((min(plotted_batch_sizes), max(plotted_batch_sizes)))
        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")

        # Reset for next axis
        plotted_batch_sizes = set()

    # fig.supxlabel("Batch size", fontsize=FONTSIZE)
    fig.supylabel("Samples / s", fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(
            handles, 
            labels, 
            loc='upper right', 
            bbox_to_anchor = (-0.22, -0.05, 1, 1), 
            bbox_transform = plt.gcf().transFigure, 
            fontsize=FONTSIZE-1
        )

    # fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(output_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if title:
        filename = f"DLIO_{workload}_{title}_throughputs.png"
    else:
        filename = f"DLIO_{workload}_throughputs.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



def plot_latencies(plotting_data, output_dir, workload, mean_or_median="median", fill_between=True, legend=False, title=None):
    print(f'Plotting latencies')

    ## Shorter plot 
    if workload == 'UNET3D':
        metrics_pretty_names = {
            "load_batch_inner": "Data Latency",
            "sample_load": "VFS Latency",
            "sample_preproc": "Sample Preprocessing",
        }
    elif workload == 'DLRM':
        metrics_pretty_names = {
            "load_batch_inner": "Data Latency",
            "batch_load": "VFS Latency",
            "batch_preproc": "Preprocessing",
        }
    else:
        metrics_pretty_names = {
            "load_batch_inner": "Data Latency",
            "batch_preproc": "Preprocessing",
        }
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }

    # Overall plot
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(max(6 * len(metrics_to_plot), 10), 5),
    )

    GPUs_to_plot = sorted(plotting_data.keys())
    plotted_batch_sizes = set()


    i_ax = 0
    for metric in metrics_to_plot:
        ax = axs[i_ax]
        i_ax += 1
        # ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

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
        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]

        # for large batch sizes
        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

            if metric == 'batch_load':
                ax.yaxis.set_major_formatter(OOMFormatter(-2, "%1.2f"))  
            if metric == 'batch_preproc':
                ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.1f"))  
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

            if metric == 'load_batch_inner':
                ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.2f"))
                ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)
            else:
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            
        ax.set_xlim((min(plotted_batch_sizes), max(plotted_batch_sizes)))

        # Reset for next axis
        plotted_batch_sizes = set()

        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
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
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(output_dir, "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if title:
        filename = f"DLIO_{workload}_{title}_latencies_{mean_or_median}{'_nofill' if fill_between is False else ''}.png"
    else:
        filename = f"DLIO_{workload}_latencies_{mean_or_median}{'_nofill' if fill_between is False else ''}.png"

    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_full_step_breakdown(data_dir, output_dir, workload, legend=False, title=None):
    metrics_to_plot_pretty_names = {
        "data_loading_throughput": "Data Throughput",
        "inner_data_loading_throughput": "Inner Data Throughput",
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

    GPUs_to_plot = GPUs_int
    batches_to_plot = batches_int

    FONTSIZE = 18

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot.keys()), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot.keys()), 5)
    )

    plotted_batch_sizes = set()

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
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]

        # for large batch sizes
        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        plotted_batch_sizes = set()

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        # ax.legend(fontsize=FONTSIZE)


    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.01, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    # fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(data_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    if title:
        filename = f"DLIO_{workload}_{title}_full_step_breakdown.png"
    else:
        filename = f"DLIO_{workload}_full_step_breakdown.png"
    figure_filename = os.path.join(output_dir, filename)
    print(f'Saved {figure_filename}')

    plt.savefig(figure_filename, format="png", dpi=450)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_step_breakdown(plotting_data, output_dir, workload, sharey=True, legend=False, title=None):

    print(f'Printing smaller plot')
    ## Shorter plot 

    metrics_pretty_names = {
        "step_end": "Overall Step",
        # "load_batch_mem": "Batch Loading",
        "load_batch_inner": "Batch Loading",
        "all_compute": "Computation",
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

    plotted_batch_sizes = set()

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

        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-3)

    # Save a reference to the legend to export later
    handles, labels = ax.get_legend_handles_labels()
    
    legend_x_offset = -0.075 if sharey else -0.06

    if legend:
        fig.legend(handles, labels, 
            loc='upper right',  
            bbox_to_anchor = (legend_x_offset, -0.05, 1, 1), 
            bbox_transform = plt.gcf().transFigure, 
            fontsize=FONTSIZE - 2
        )

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    # fig.supxlabel('Batch size', fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    output_dir = os.path.join(output_dir, "plots")
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if title:
        filename = f"DLIO_{workload}_{title}_step_breakdown{'_sharey' if sharey else ''}.png"
    else:
        filename = f"DLIO_{workload}_step_breakdown{'_sharey' if sharey else ''}.png"
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    plt.cla() 
    plt.close('all')   
    plt.close(fig)


    # Export just the legend
    fig = plt.figure(figsize=(2,3))

    fig.legend(handles, labels,
        fontsize=FONTSIZE
    )

    filename = f"DLIO_{workload}_step_breakdown{'_sharey' if sharey else ''}_LEGEND.png"
    figure_filename = os.path.join(output_dir, filename)
    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    plt.cla() 
    plt.close('all')   
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate average times spent in diff phases of training")
    parser.add_argument("data_dirs", nargs='+', help="Data directories")
    parser.add_argument("workload", help="Workload", choices=['unet3d', 'bert', 'dlrm'])
    parser.add_argument("-o", "--output-dir", help="Output directory", default='data_step_breakdown')
    parser.add_argument("-t", "--title", help="Title for plots", default=None)
    parser.add_argument("-l", "--legend", action="store_true", help="Add the legend", default=None)
    args = parser.parse_args()

    data_dirs = args.data_dirs
    output_dir = args.output_dir
    workload = args.workload.upper()
    title = args.title
    legend = args.legend

    # In the usual case of a single data directory, output in it
    if len(data_dirs) == 1:
        output_dir = args.data_dirs[0]
    else:
        if args.output_dir is None:
            output_dir = f"data_step_breakdown/DLIO_{workload}_{int(time.time())}"
        else:
            output_dir = args.output_dir
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)


    plotting_data = generate_plotting_data(data_dirs, output_dir, workload)

    plot_throughputs(plotting_data, output_dir, workload, title=title, legend=legend)
    plot_latencies(plotting_data, output_dir, workload, mean_or_median="median", title=title, legend=legend)

    exit()
    plot_step_breakdown(plotting_data, output_dir, workload, title=title, legend=legend)
    # plot_step_breakdown(data_dir, output_dir, workload, sharey=False, title=title)

    # plot_data_loading(plotting_data, workload, mean_or_median="median", title=title)
    # plot_data_loading(plotting_data, workload, mean_or_median="mean", fill_between=False, title=title)

    # plot_sample_load(plotting_data, output_dir, workload, mean_or_median="mean", title=title)
    # plot_sample_load(plotting_data, output_dir, workload, mean_or_median="mean", fill_between=False, title=title)
