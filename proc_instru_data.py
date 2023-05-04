import os
import re
import json
import copy
import time
import argparse
import numpy as np
from pathlib import Path
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter


from scipy.stats import norm, linregress

from sklearn.linear_model import LinearRegression

from proc_instru_data_dlio import OOMFormatter


METRICS_PER_WORKLOAD = {
    "UNET3D": {
        "step_end": "Overall step",
        "load_batch_mem": "1 Batch load to mem",
        "all_compute": "2 Computation",
        "sample_load": "Sample load",
        "sample_preproc": "Sample preprocessing",
    },
    "DLRM": {
        "step_end": "Overall step",
        "load_batch_mem": "1 Batch load to mem",
        "all_compute": "2 Computation",
        "batch_load": "Batch from disk",
        "batch_preproc": "Batch preproc",
    }
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

FONTSIZE = 24


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
        res = re.search(r'.*_([0-9]+)b.*', log_file_name)
    if res is None:
        res = re.search(r'.*b([0-9]+).*', log_file_name)
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

def get_all_experiment_log_files(data_dirs):
    log_files = []
    for data_dir in data_dirs:
        log_files.extend([os.path.join(data_dir, 'raw_data', f) for f in os.listdir(os.path.join(data_dir, 'raw_data'))])
    
    return sorted(log_files)


def preprocess_data(data_dirs, output_dir, workload, fit=False, big_histo=False, title=None):

    output_dir = Path(output_dir)

    log_files = get_all_experiment_log_files(data_dirs)

    metrics_pretty_names = METRICS_PER_WORKLOAD[workload]
    events_of_interest = set(metrics_pretty_names)

    all_metrics = { metric: [] for metric in metrics_pretty_names }
    all_metrics['data_loading_throughput'] = []
    all_metrics['data_proc_throughput'] = []
    all_metrics['from_disk_throughput'] = []
    all_metrics['step_throughput'] = []

    # DS for plotting data
    all_data = {}
    fit_data = {}

    all_gpus = set()
    all_batches = set()

    for log_file in log_files:
        # print(f'Processing {log_file}')

        gpu_key = get_num_gpus(log_file)
        batch_key = get_batch_size(log_file)

        all_gpus.add(gpu_key)
        all_batches.add(batch_key)

        if gpu_key not in all_data:
            all_data[gpu_key] = {
                batch_key: { metric: [] for metric in all_metrics }
            }
            fit_data[gpu_key] = {
                batch_key: {
                    'mu': [],
                    'std': [],
                }
            }
        else:
            if batch_key not in all_data[gpu_key]: 
                all_data[gpu_key][batch_key] = { metric: [] for metric in all_metrics }
            if batch_key not in fit_data[gpu_key]: 
                fit_data[gpu_key][batch_key] = {
                    'mu': [],
                    'std': [],
                }


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
                if title != 'sleep' and seen_events != events_of_interest:
                    seen_events.add(event)
                    continue

                # Append value to appropriate array
                value = round(line['value']['duration'] / 1_000_000_000, 6)
                if value == 0:
                    continue

                # Define data loading bandwidth as per worker batch size / time to load batch
                if event == 'load_batch_mem':
                    all_data[gpu_key][batch_key]['data_loading_throughput'].append(batch_key / value)

                if event == 'all_compute':
                    all_data[gpu_key][batch_key]['data_proc_throughput'].append(batch_key / value)
                
                if event == 'sample_load':
                    all_data[gpu_key][batch_key]['from_disk_throughput'].append(1 / value)
                
                if event == 'batch_load':
                    all_data[gpu_key][batch_key]['from_disk_throughput'].append(batch_key / value)

                if event == 'step_end':
                    all_data[gpu_key][batch_key]['step_throughput'].append(batch_key / value)

                all_data[gpu_key][batch_key][event].append(value)

        if fit:
            # Once we've gone through the whole log file, plot a histogram of the computation time
            # and fit a gaussian to it (visual inspection reveals it looks normally distributed)
            mu, std = fit_normal_distrib_and_plot(all_data[gpu_key][batch_key]['all_compute'], output_dir, title=f"all_compute_{gpu_key}g_{batch_key}b")
            fit_data[gpu_key][batch_key]['mu'].append(mu)
            fit_data[gpu_key][batch_key]['std'].append(std)

    # Save all data
    with open(output_dir / "all_data.json", "w") as alldatafile:
        json.dump(all_data, alldatafile, indent=4)

    # Save the fit data
    if fit:
        with open(output_dir / "compute_time_fit.json", "w") as outfile:
            json.dump(fit_data, outfile, indent=4)

        filename = Path(output_dir) / 'ditribution_fits.txt'

        with open(filename, 'w') as outfile:
            fit_linear_to_distribs(fit_data, outfile)



    if big_histo:
        # Plot a super histogram with all the fits
        all_batches = sorted(list(all_batches), key=int)
        all_gpus = sorted(list(all_gpus), key=int)

        fig, axes = plt.subplots(ncols=len(all_gpus), nrows=len(all_batches), figsize=(2 + 2*len(all_gpus), 1 + 2 * len(all_batches)))

        row_headers = [f'Batch size: {b}' for b in all_batches]
        col_headers = [f'{g} GPU{"s" if g > 1 else ""}' for g in all_gpus]

        font_kwargs = dict(fontsize="large")
        add_headers(fig, col_headers=col_headers, row_headers=row_headers, **font_kwargs)

        fig.tight_layout()
        # fig.suptitle('UNET3D computation time distribution')

        supblot_xmins = []
        supblot_xmaxs = []

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
                xmin, xmax = single_histogram(ax, all_data[gpu][batch]['all_compute'])

                supblot_xmins.append(xmin)
                supblot_xmaxs.append(xmax)

        # plt.setp(axes, xlim=(0, 0.3))

        print(min(supblot_xmins), max(supblot_xmaxs))
        # Create output directory if it doesn't exist
        outdir = Path(output_dir) / "histograms"
        outdir.mkdir(parents=True, exist_ok=True)

        figure_filename = outdir / f'{workload}_all_histograms.png'

        plt.savefig(figure_filename, format="png", dpi=500)
        print(f'Saved {figure_filename}')
        # Clear the current axes.
        plt.cla() 
        # Closes all the figure windows.
        plt.close('all')   
        plt.close(fig)
  
    # Deepcopy all_data but we'll replace the arrays with dictionaries of summary stats
    plotting_data = copy.deepcopy(all_data)

    all_evals = []
    with open(output_dir / f"{workload}_step_analysis.log", "w") as outfile:
        # Print header
        outfile.write(f"{'Metric':>30}\t{'Mean':>15}\t{'Std':>15}\t{'q1':>15}\t{'Median':>15}\t{'q3':>15}\n")

        for gpu_key in all_data:
            for batch_key in all_data[gpu_key]:
                
                outfile.write(f"{workload}_{gpu_key}GPU_{batch_key}batch\n")

                # all_evals.extend(all_data[gpu_key][batch_key]['eval_sliding_window'])

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

    return plotting_data, all_data



def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386
    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def single_histogram(axis, data, nbins=200):

    data = sorted(data)
    p1 = np.percentile(data, 1.0)
    p99 = np.percentile(data, 99.0)

    # Remove p1 and p99 outliers from data
    data = [x for x in data if p1 <= x <= p99]

    mu, std = norm.fit(data)

    print(f'num data: {len(data)}, p99: {p99}')
    print(f'mu: {mu}, std: {std}')

    data = np.asarray(data)

    axis.hist(data, bins=nbins, density=True)

    xmin, xmax = axis.get_xlim()
    x = np.linspace(xmin, xmax, len(data))

    p = norm.pdf(x, mu, std)
    axis.plot(x, p, 'k', linewidth=1)

    axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axis.tick_params(axis='both', which='major', labelsize=9)
    axis.axvline(mu, color='k', linestyle='dashed', linewidth=1)

    # place a text box in upper left in axes coords
    axis.text(0.6, 0.95, f"mu={mu:.3f}\nstd={std:.3f}\nN={len(data)}", transform=axis.transAxes, fontsize=9,
        verticalalignment='top')
    
    return xmin, xmax




def fit_linear_to_distribs(data, outfile):

    batches = []
    gpus = []
    mus = []
    stds = []
    for gpu in data:
        for batch in data[gpu]:
            mus.extend(data[gpu][batch]['mu'])
            stds.extend(data[gpu][batch]['std'])
            batches.extend([batch] * len(data[gpu][batch]['mu']))
            gpus.extend([gpu] * len(data[gpu][batch]['mu']))

    batches = np.asarray(batches)
    gpus = np.asarray(gpus)
    mus = np.asarray(mus)
    stds = np.asarray(stds)

    X = np.column_stack((gpus, batches))

    print(X)

    reg = LinearRegression().fit(X, mus)
    R2 = reg.score(X, mus)

    print(f'''Fitting lin reg gpus, batches and mus:
    Model: 
    \tcompute_time_mean = np.dot({reg.coef_}, [num_gpus, batch_size]) + {reg.intercept_}
    \tR2: {R2}''')
    outfile.write(f'''Fitting lin reg gpus, batches and mus:
    \tModel: 
    \t\tcompute_time_mean = np.dot({reg.coef_}, [num_gpus, batch_size]) + {reg.intercept_}
    \tR2: {R2}\n''')

    reg = LinearRegression().fit(X, stds)
    R2 = reg.score(X, stds)

    print(f'''Fitting lin reg gpus, batches and stds:
    Model:
    \t\tcompute_time_std = np.dot({reg.coef_}, [num_gpus, batch_size]) + {reg.intercept_}
    \tR2: {R2}''')
    outfile.write(f'''Fitting lin reg gpus, batches and stds:
    Model:
    \tcompute_time_std = np.dot({reg.coef_}, [num_gpus, batch_size]) + {reg.intercept_}
    R2: {R2}\n''')


def fit_normal_distrib_and_plot(data, output_dir, title='histo', nbins=200):

    data = sorted(data)
    p1 = np.percentile(data, 1.0)
    p99 = np.percentile(data, 99.0)
    # Remove p1 and p99 outliers from data
    data = [x for x in data if p1 <= x <= p99]

    mu, std = norm.fit(data)

    print(f'num data: {len(data)}, p99: {p99}')
    print(f'mu: {mu}, std: {std}')

    return mu, std


def plot_histogram(all_data, metric, output_dir, gpus=None, batches=None, title='histo', nbins=200):
    data = []

    for gpu in all_data:
        if gpus is not None and gpu not in gpus:
            continue
        for batch in all_data[gpu]:
            if batches is not None and batch not in batches:
                continue
            data.extend(all_data[gpu][batch][metric])

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(15,15))

    with open(Path(output_dir) / "preproc_times.json", "w") as outfile:
        json.dump(data, outfile)

    data = np.asarray(data)
    from collections import Counter
    counts = Counter(data)

    
    print(counts.most_common(10))

    ax.hist(data, bins=nbins)

    median = np.median(data)
    trans = ax.get_xaxis_transform()
    ax.axvline(median, color='k', linestyle='dashed', linewidth=1)
    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.text(median * 1.5, .85, f'median: {median:2.4f}', transform=trans)

    # Create output directory if it doesn't exist
    outdir = Path(output_dir) / "histograms"
    outdir.mkdir(parents=True, exist_ok=True)

    figure_filename = outdir / f'{metric}_{title}.png'

    plt.savefig(figure_filename, format="png", dpi=250)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



def plot_step_breakdown(plotting_data, output_dir, workload, sharey=True, fill_between=True, legend=False, title=None):
    print(f'Plotting Step breakdown summary')

    ## Shorter plot 
    metrics_pretty_names = {
        # "eval_sliding_window": "Eval Sliding Window",
        "step_end": "Overall Step",
        "load_batch_mem": "Data Loading",
        "all_compute": "Computation",
    }

    if workload == 'UNET3D':
        metrics_pretty_names = {
            "step_end": "Overall Step",
            "load_batch_mem": "Data Loading",
            "load_batch_mem2": "Data Loading (no fill)",
            "all_compute": "Computation",
        }
        FONTSIZE = 18
    else:
        FONTSIZE = 16

    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }
    
    output_dir = os.path.join(output_dir, "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    # Overall plot
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(3.1 * len(metrics_to_plot), 4), 
        sharey=sharey
    )
    # if not isinstance(axes, list):
    #     axes = [axes]
    GPUs_to_plot = sorted(plotting_data.keys())
    plotted_batch_sizes = set()

    i_ax = -1

    for metric in metrics_to_plot:
        i_ax += 1
        ax = axes[i_ax]

        if metric == 'load_batch_mem2':
            ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)
            metric = 'load_batch_mem'
            fill_between = False
        else:
            fill_between = True
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
            
            ax.plot(x, y, label=f"{gpu_key} GPU{'s' if gpu_key > 1 else ''}", )

            if fill_between:
                q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
                q1 = np.asarray(q1)

                q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
                q3 = np.asarray(q3)
                
                ax.fill_between(x, q1, q3, alpha=0.1)

        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]
        
        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.7, color="grey")
        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.set_xlim((min(plotted_batch_sizes), max(plotted_batch_sizes)))

        plotted_batch_sizes = set()


    handles, labels = ax.get_legend_handles_labels()

    if legend:
        fig.legend(
            handles, 
            labels, 
            loc='upper right',
            bbox_to_anchor = (-0.065, -0.05, 1, 1), 
            bbox_transform = plt.gcf().transFigure, 
            fontsize=FONTSIZE - 2
        )

    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel('Batch Size', fontsize=FONTSIZE)

    if title:
        filename = f"{workload}_{title}_step_breakdown{'_sharey' if sharey else ''}{'_nofill' if fill_between is False else ''}.png"
    else:    
        filename = f"{workload}_step_breakdown{'_sharey' if sharey else ''}{'_nofill' if fill_between is False else ''}.png"

    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def plot_throughputs(plotting_data, output_dir, workload, legend=False, title=None):
    print('Plotting Throughputs')

    metrics_to_plot_pretty_names = {
        # 'step_throughput': 'Step Throughput',
        "from_disk_throughput": "VFS Throughput",
        "data_loading_throughput": "Data Throughput",
        "data_proc_throughput": "Compute Throughput",
    }
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names }

    output_dir = os.path.join(output_dir, "plots")
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    GPUs_to_plot = sorted(plotting_data.keys())
    # GPUs_to_plot = [4,6,8]
    plotted_batch_sizes = set()

    # Overall plot
    fig, axes = plt.subplots(nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        # figsize=(max(6 * len(metrics_to_plot), 10), 6),
        figsize=(max(6 * len(metrics_to_plot), 10), 5),
    )

    min_batch = float('inf')
    max_batch = 0

    i_ax = -1
    for metric in metrics_to_plot:
        i_ax += 1
        ax = axes[i_ax]
        if metric == 'data_loading_throughput2':
            metric = 'data_loading_throughput'

        if title not in ['gen', 'sleep']:
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
            ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

            if metric == 'from_disk_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(7, "%1.1f")) 

            if metric == 'data_loading_throughput':
                if title in [None, 'real']:
                    ticks = [1.0, 2.0, 3.0, 4.0, 5.0]
                    tick_labels = [str(t) for t in ticks]
                    ticks = [1e6, 2e6, 3e6, 4e6, 5e6]
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(tick_labels)

                ax.yaxis.set_major_formatter(OOMFormatter(6, "%1.1f"))
            
            if metric == 'data_proc_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(5, "%1.1f"))  

        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

            if metric == 'data_proc_throughput':
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

            if metric == 'data_loading_throughput':
                ax.yaxis.set_major_formatter(OOMFormatter(3, "%2.0f"))
                ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

        ax.set_xlim((min(plotted_batch_sizes), max(plotted_batch_sizes)))

        # Reset for next axis
        plotted_batch_sizes = set()

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        # ax.set_xlabel("Batch size", fontsize=FONTSIZE)
        # ax.set_ylabel("Samples / s", fontsize=FONTSIZE)


    # fig.supxlabel("Batch size", fontsize=FONTSIZE)
    fig.supylabel("Samples / s", fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()
    if legend:
        fig.legend(handles, labels, loc='upper right',  bbox_to_anchor = (-0.25, -0.05, 1, 1), bbox_transform = plt.gcf().transFigure, fontsize=FONTSIZE-1)

    if title:
        filename = f"{workload}_{title}_throughputs.png"
    else:
        filename = f"{workload}_throughputs.png"

    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')

    plt.cla() 
    plt.close('all')   
    plt.close(fig)







def plot_latencies(plotting_data, output_dir, workload, mean_or_median="median", fill_between=True, legend=False, title=None):
    print(f'Plotting latencies')

    metrics_pretty_names = {}

    if workload == 'UNET3D':
        metrics_pretty_names["load_batch_mem"] = "Data Latency"
        metrics_pretty_names["sample_load"] = "VFS Latency"
        metrics_pretty_names["sample_preproc"] = "Sample Preprocessing"
        
    if workload == 'DLRM':
        metrics_pretty_names["load_batch_mem"] = "Data Latency"
        metrics_pretty_names["batch_load"] = "VFS Latency"
        metrics_pretty_names["batch_preproc"] = "Batch Preprocessing"
    
    metrics_to_plot = { metric: [] for metric in metrics_pretty_names }

    output_dir = os.path.join(output_dir, "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Overall plot
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=len(metrics_to_plot), 
        layout="constrained", 
        figsize=(max(6 * len(metrics_to_plot), 10), 5),
        sharey=False
    )

    GPUs_to_plot = sorted(plotting_data.keys())
    plotted_batch_sizes = set()


    i_ax = 0
    for metric in metrics_to_plot:
        ax = axs[i_ax]
        i_ax += 1

        if title not in ['gen', 'sleep']:
            ax.set_title(metrics_pretty_names[metric], fontsize=FONTSIZE)

        # plot the metric in the axes
        for gpu_key in GPUs_to_plot:

            batches_to_plot = sorted(list(plotting_data[gpu_key].keys()))
            plotted_batch_sizes.update(batches_to_plot)

            x = np.asarray(batches_to_plot)

            y = [ plotting_data[gpu_key][batch][metric][mean_or_median] for batch in batches_to_plot ]
            y = np.asarray(y)

            ax.plot(x, y, label=f"{gpu_key} GPU{'s' if gpu_key > 1 else ''}", )

            # if fill_between and not (metric == 'load_batch_mem' and workload == 'UNET3D'):
            if fill_between:
                if mean_or_median == "median":
                    q1 = [ plotting_data[gpu_key][batch][metric]["q1"] for batch in batches_to_plot ]
                    q1 = np.asarray(q1)

                    q3 = [ plotting_data[gpu_key][batch][metric]["q3"] for batch in batches_to_plot ]
                    q3 = np.asarray(q3)
                    
                    ax.fill_between(x, q1, q3, alpha=0.1)
                else:
                    std = [ plotting_data[gpu_key][batch][metric]["std"] for batch in batches_to_plot ]
                    std = np.asarray(std)

                    ax.fill_between(x, y - std, y + std, alpha=0.1)

        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]
        
        # ax.yaxis.set_major_formatter('{x:0<4.2f}')

        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.offsetText.set_fontsize(FONTSIZE-2)

            if metric == 'load_batch_mem':
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
                if title == 'gen':
                    ax.locator_params(axis='y', nbins=3)

            if metric == 'batch_load':
                ax.yaxis.set_major_formatter(OOMFormatter(-2, "%1.2f"))
            if metric == 'batch_preproc':
                ax.yaxis.set_major_formatter(OOMFormatter(-3, "%1.2f"))  

        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

            if title == 'gen' and metric == 'load_batch_mem':
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

        ax.set_xlim((min(plotted_batch_sizes), max(plotted_batch_sizes)))

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

    if title:
        filename = f"{workload}_{title}_latencies_{mean_or_median}{'_nofill' if fill_between is False else ''}.png"
    else:
        filename = f"{workload}_latencies_{mean_or_median}{'_nofill' if fill_between is False else ''}.png"

    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)




def plot_full_breakdown(plotting_data, output_dir, workload, sharey=True, legend=False, title=None):
    print(f'Plotting full step breakdown (sharey={sharey})')

    metrics_to_plot_pretty_names = {
        "step_end": "Overall",
        "load_batch_mem": "1 Data Loading",
        "all_compute": "2 Computation",
        "load_batch_gpu": "Batch load to GPU",
        "model_forward_pass": "Forward pass",
        "loss_tensor_calc": "Loss calculation",
        "model_backward_pass": "Backward pass",
        "model_optim_step": "Optimizer step", 
        "cum_loss_fn_calc": "Cumulative loss",
    }
    if workload == 'UNET3D':
        metrics_to_plot_pretty_names["cum_loss_fn_calc"] = "Cumulative loss"

    metrics_present_in_data = set()

    for gpu_key in plotting_data:
        for batch_key in plotting_data[gpu_key]:
            metrics_present_in_data = list(plotting_data[gpu_key][batch_key].keys())
            break
        break

    # print(metrics_present_in_data)
    metrics_to_plot = { metric: [] for metric in metrics_to_plot_pretty_names if metric in metrics_present_in_data}

    GPUs_to_plot = sorted(plotting_data.keys())
    plotted_batch_sizes = set()

    FONTSIZE = 15

    output_dir = os.path.join(output_dir, "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Overall plot
    # fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(3.1 * len(metrics_to_plot.keys()), 5), sharey=sharey)
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics_to_plot.keys()), layout="constrained", figsize=(2.7 * len(metrics_to_plot.keys()), 4), sharey=sharey)

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
            
            ax.plot(x, y, label=f"{gpu_key} GPU{'s' if gpu_key > 1 else ''}")

            ax.fill_between(x, q1, q3, alpha=0.1)
        
        plotted_batch_sizes = sorted(list(plotted_batch_sizes))
        plotted_batch_sizes_int = [ int(b) for b in plotted_batch_sizes]

        if workload == 'DLRM':
            ax.set_xscale('log', base=2)
            x_axis_labels = [DLRM_BATCH_SIZE_STRINGS[b] for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, rotation=-35, ha='center', fontsize=FONTSIZE-5)
        else:
            x_axis_labels = [str(b) for b in plotted_batch_sizes]
            ax.set_xticks(plotted_batch_sizes_int, x_axis_labels, ha='center', fontsize=FONTSIZE-2)

        ax.tick_params(which="both", direction="in", labelsize=FONTSIZE-1)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.4, color="grey")
        plotted_batch_sizes = set()


    fig.supylabel("Time (s)", fontsize=FONTSIZE)
    fig.supxlabel("Batch size", fontsize=FONTSIZE)

    handles, labels = ax.get_legend_handles_labels()

    if legend:
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
    
    if title:
        filename = f"{workload}_{title}_full_breakdown{'' if sharey else '_indiv'}.png"
    else:
        filename = f"{workload}_full_breakdown{'' if sharey else '_indiv'}.png"
        
    figure_filename = os.path.join(output_dir, filename)

    plt.savefig(figure_filename, format="png", dpi=450)
    print(f'Saved {figure_filename}')
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate step breakdowns, throughputs and latencies plots from UNET3D and DLRM instrumentation data.")
    parser.add_argument("data_dirs", nargs='+', help="Data directories")
    parser.add_argument("workload", help="Workload", choices=['unet3d', 'dlrm'])
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory. Defaults to the data directory for single dir and 'data_step_breakdown' for multiple data directories.")
    parser.add_argument("-t", "--title", default=None, help="Additonal string to put after workload name for plots")
    parser.add_argument("-l", "--legend", action="store_true", help="Add legend to plots")
    parser.add_argument("-f", "--fit", action="store_true", help="Fit model to distributions or not")
    parser.add_argument("-pb", "--breakdown", action="store_true", help="Plot the step breakdown.")
    parser.add_argument("-pt", "--throughputs", action="store_true", help="Plot the throughputs.")
    parser.add_argument("-pl", "--latencies", action="store_true", help="Plot the latencies.")
    parser.add_argument("-bh", "--big-histo", action="store_true", help="Save file with all compute time distributions and fits for the annex.")
    args = parser.parse_args()


    if not (args.breakdown or args.throughputs or args.latencies):
        print('No type of plot requested. Exiting.')
        exit()

    data_dirs = args.data_dirs
    workload = args.workload.upper()
    title = args.title
    fit = args.fit
    big_histo = args.big_histo
    legend = args.legend

    # In the usual case of a single data directory, output in it
    if len(data_dirs) == 1:
        output_dir = args.data_dirs[0]
    else:
        if args.output_dir is None:
            output_dir = f"data_step_breakdown/{workload}_{int(time.time())}"
        else:
            output_dir = args.output_dir
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    plotting_data, all_data = preprocess_data(data_dirs, output_dir, workload, fit=fit, big_histo=big_histo, title=title)

    if args.breakdown:
        plot_step_breakdown(plotting_data, output_dir, workload, sharey=False, title=title, legend=legend)

    if args.throughputs:
        plot_throughputs(plotting_data, output_dir, workload, title=title, legend=legend)

    if args.latencies:
        plot_latencies(plotting_data, output_dir, workload, title=title, legend=legend)



