import os
import re
import json
import pathlib
import argparse
import numpy as np
import pandas as pd
from os.path import isfile, isdir, join
from matplotlib import dates as mdates, pyplot as plt, patches as mpatches


def plot_all_configs(data_dir, workload, title, all_plots=False, **kwargs):
    """
    Create a timeline plot for every configuration present under data_dir/timeline/pid.
    Create a 'paper format' plot using the all_combined config, which should always exist.
    """

    # Each timeline configuration has a directory under data_dir/pid
    timeline_dirs = [f.path for f in os.scandir(join(data_dir, 'pid')) if f.is_dir()]

    for timeline_dir in timeline_dirs:

        plot_pids_timeline_cpu_gpu(data_dir, timeline_dir, workload, f'{title} Overview', name='overview', **kwargs)

        if all_plots:

            interesting_time_ranges = get_plotting_ranges(args.data_dir, args.workload)

            for zoom_name, time_range in interesting_time_ranges.items():
                if time_range is None:
                    continue
                
                print(f'Zooming into {zoom_name}')
                plot_pids_timeline_cpu_gpu(
                    data_dir,
                    timeline_dir,
                    workload,
                    title = f"{title} {zoom_name}",
                    # filename = f"{title} {zoom_name}",
                    name = zoom_name,
                    start = time_range[0],
                    end = time_range[1],
                    xformat = "%H:%M:%S",
                    margin = np.timedelta64(1, "s") if zoom_name != "init" else np.timedelta64(100, "ms"),
                )

    # For the paper version, plot the all combined config
    combined_config_dir = join(data_dir, 'pid/all_combined')
    plot_pids_timeline_cpu_gpu(data_dir, combined_config_dir, workload, title, paper_version=True, **kwargs)



def _verify_all_necessary_data_present(data_dir) -> bool:
    """
    Returns true if all essential data is present.
    """
    all_traces = ['gpu_avg.csv', 'cpu_all.csv', 'timeline.csv']

    success = True
    for trace in all_traces:
        expected_filename = join(data_dir, trace)

        if not isfile(expected_filename):
            print(f'ERROR: Missing essential trace {expected_filename}')
            success = False
    
    if not isdir(join(data_dir, 'pid')):
        print(f'ERROR: Missing essential trace {expected_filename}')
        success = False
    else:
        plot_config_dirs = [f.path for f in os.scandir(join(data_dir, 'pid')) if f.is_dir()]
        for dir in plot_config_dirs:
            if not isfile(join(dir, 'plotting_info.json')):
                print(f'ERROR: Missing plotting_info.json in {dir}')
                success = False

    return success


def _get_plotting_info_json(timeline_dir):
    # Maps data files for timelines to pretty name
    with open(join(timeline_dir, "plotting_info.json"), 'r') as infile:
        return json.load(infile)



def plot_cpu_info(data_dir, ax, fontsize=16, start=None, end=None) -> None:
    """
    Plots the CPU information on the given axis
    """
    df = pd.read_csv(
        os.path.join(data_dir, "cpu_all.csv"),
        sep=",",
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= np.datetime64(start)]
    if end is not None:
        df = df[df["timestamp"] <= np.datetime64(end)]

    ax.set_title("CPU Usage", fontsize = fontsize + 2)
    ax.set_ylabel("Percent Use (%)", fontsize = fontsize)

    # There are more fields available but weren't very interesting
    variables = [
        "%usr",
        "%sys",
        "%iowait",
        "%idle",
    ]

    n_features = len(variables)

    cm = plt.get_cmap("gist_rainbow")  # Colormap

    for i, var in enumerate(variables):
        line = ax.plot(df["timestamp"], df[var], label=var, linewidth=2)
        line[0].set_color(cm(1 * i / n_features))

    ax.grid(True, which="both", linestyle="--", color="grey", alpha=0.2)
    ax.tick_params(which="both", direction="in", labelsize=fontsize)
    ax.set_ylim(ymin=0)
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center right", fontsize=fontsize)


def plot_gpu_info(data_dir, ax1, fontsize=16, start=None, end=None):
    """
    Plots GPU info on given axis.
    """

    df = pd.read_csv(os.path.join(data_dir, "gpu_avg.csv"), sep=",", on_bad_lines='skip')
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if start is not None:
        df = df[df["timestamp"] >= np.datetime64(start)]
    if end is not None:
        df = df[df["timestamp"] <= np.datetime64(end)]

    ax1.set_title("GPU Usage", fontsize=fontsize+2)
    ax1.set_ylabel("Usage (%)", fontsize=fontsize)

    ax1.plot(
        df["timestamp"],
        df["sm"],
        label="GPU MultiProcessor Use (%)",
        color="tab:red",
        linewidth=2,
        markersize=5,
    )
    ax1.plot(
        df["timestamp"],
        df["mem"],
        label="GPU Memory Use (%)",
        color="tab:orange",
        linewidth=2,
        markersize=5,
    )

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  

    ax2.set_ylabel("Size (MB)", fontsize=fontsize)
    ax2.plot(
        df["timestamp"],
        df["fb"],
        label="Framebuffer Memory Use (MB)",
        color="tab:blue",
        linewidth=2,
        markersize=5,
        rasterized=True
    )

    ax1.grid(True, which="both", linestyle="--")
    ax1.tick_params(which="both", direction="in", grid_color="grey", grid_alpha=0.3, labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)

    ax1.set_ylim(ymin=0, ymax=100)
    ax2.set_ylim(ymin=0)

    # This will combine the GPU %mp, %mem and FBmem legends  
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    ax2.legend(handles, labels, loc='center right', fontsize=fontsize)


def plot_iostat_info(data_dir, ax, fontsize=16, start=None, end=None):
    """
    Plots the iostat info.
    """

    df = pd.read_csv(
            join(data_dir, "iostat.csv"),
            sep=",",
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if start is not None:
        df = df[df["timestamp"] >= np.datetime64(start)]    
    if end is not None:
        df = df[df["timestamp"] <= np.datetime64(end)]

    colormap = {
        "rMB/s": "blue", 
        "wMB/s": "red"
    }
    variables = colormap.keys()

    # Find max value of our variables for either disk and use it as ymax
    # We'll use the same ymax for all disk plots to visually compare them
    ymax = df[variables].max().max()

    # Combine disk data into a single plot
    ax.set_title(f"Disk Usage (iostat)", fontsize = fontsize + 2)
    ax.set_ylabel("Usage (MB/s)", fontsize = fontsize)

    df2 = df.groupby('timestamp').sum()
    for i, var in enumerate(variables):
        line = ax.plot(df2.index, df2[var], label=var, linewidth=2)
        line[0].set_color(colormap[var])

    ax.grid(True, which="both", linestyle="--", color="grey", alpha=0.2)
    ax.tick_params(which="both", direction="in", labelsize = fontsize)

    ax.set_ylim(ymin=0, ymax=ymax)
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left", fontsize=fontsize)


def plot_trace_timeline(timeline_dir, timeline_file, plotting_info, ax, timeline_config, plot_legend=False, fontsize=16, start=None, end=None, margin=None):
    """
    Plots the BPF traces onto a timeline.
    """
    print(f"Plotting timeline {os.path.basename(timeline_file)}")

    bar_height = timeline_config['bar_height']
    ymins = timeline_config['ymins']
    categories = timeline_config['categories']
    colors_dict = timeline_config['colors_dict']

    ptitle = plotting_info[timeline_file]
    
    ax.set_title(f"{ptitle}", fontsize=fontsize+2)

    df = pd.read_csv(
        join(timeline_dir, timeline_file), names=["start_date", "end_date", "event"]
    )
    df.start_date = pd.to_datetime(df.start_date).astype(np.datetime64)
    df.end_date = pd.to_datetime(df.end_date).astype(np.datetime64)

    if start is not None:
        df = df[df["start_date"] >= np.datetime64(start)]
    if end is not None:
        df = df[df["end_date"] <= np.datetime64(end)]

    # If the DataFrame is empty after filtering, skip
    if len(df) == 0:
        print(f"This timerange is empty for pid {timeline_file}. Skipping.")
        return

    # Can't define this earlier
    masks = {
        "BIO": (df["event"] == "BIOR") | (df["event"] == "BIOW"),
        "OPEN": (df["event"] == "OPENAT"),
        "R/W": (df["event"] == "READ") | (df["event"] == "WRITE"),
    }

    # Plot the events
    for j, category in enumerate(categories):
        mask = masks[category]
        start_dates = mdates.date2num(df.loc[mask].start_date)
        end_dates = mdates.date2num(df.loc[mask].end_date)
        durations = end_dates - start_dates
        xranges = list(zip(start_dates, durations))
        ymin = ymins[j] - 0.5
        yrange = (ymin, bar_height)
        colors = [colors_dict[event] for event in df.loc[mask].event]
        ax.broken_barh(xranges, yrange, facecolors=colors, alpha=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.grid(True, axis="both", linestyle="--", linewidth=0.45, alpha=0.2, color="grey")
    ax.tick_params(which="both", direction="in", labelsize=fontsize)

    # Format the y-ticks
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    # Add the legend
    if plot_legend:
        patches = [
            mpatches.Patch(color=color, label=key) for (key, color) in colors_dict.items()
        ]
        ax.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left", fontsize=fontsize)

    # Sometimes the range we try to plot contains nothing so the limits are NaT
    # and the program throws a value error "Axis limits cannot be NaN or Inf"
    try:
        ax.set_xlim(
            df.start_date.min() - margin,
            df.end_date.max() + margin,
        )
    except Exception as e:
        print(f"Exception caught while trying to set graph limits: {e}")
        print("Skipping this graph.")
        return


def plot_mllog_events(data_dir, ax, plot_config, fontsize=16, name=None, start=None, end=None, xformat='%H:%M', vlines=None):
    """
    Plots the timeline of MLLOG events.
    The specific events present will depend on the workload.
    """
    print(f"Processing timeline")

    bar_height = plot_config['bar_height']
    ymins = plot_config['ymins']
    categories = plot_config['categories']
    colors_dict = plot_config['colors_dict']


    df = pd.read_csv(join(data_dir, "timeline.csv"), names=["start_date", "end_date", "event"])
    df.start_date = pd.to_datetime(df.start_date).astype(np.datetime64)
    df.end_date = pd.to_datetime(df.end_date).astype(np.datetime64)

    if start is not None:
        print(f"Filtering with start date >= {start}")
        df = df[df["start_date"] >= np.datetime64(start)]
        print(df.head())
    if end is not None:
        print(f"Filtering with end date <= {end}")
        df = df[df["end_date"] <= np.datetime64(end)]
        print(df.head())

    # Add synthetic data to show timeline info when no data point is included in the desired range
    # Uncomment/modify according to needs

    # if df.shape[0] == 0:
    #     print("empty will create default data")
    #     df.loc[-1] = [ np.datetime64(start),  np.datetime64(end), "EPOCH"]
    # # elif df.shape[0] == 1:

    # if title == "MLCommons Image Segmentation - 4 GPUs 1xRAM dataset - Naive Copy First 5 Min":
    #     init_period = df.iloc[0]
    #     print("pad end with epoch")
    #     df.loc[-1] = [ init_period["end_date"],  np.datetime64(end), "EPOCH"]
    #     print(df)
    #     print(df.shape)
    # elif title == "MLCommons Image Segmentation - 4 GPUs 1xRAM dataset - First Eval":
    #     print("Pad training with epochs")
    #     eval_period = df.iloc[0]
    #     eval_start = eval_period["start_date"] - np.timedelta64(1, "us")
    #     df.loc[-1] = [ np.datetime64(start),  eval_start, "EPOCH"]
    #     eval_stop = eval_period["end_date"] + np.timedelta64(1, "us")
    #     df.loc[-2] = [ eval_stop,  np.datetime64(end), "EPOCH"]
    #     print(df)
    #     print(df.shape)
    # else:
    if df.shape[0] == 0:
        if name == 'first_30s' or name == 'init':
            print("Pad with init")
            df.loc[-1] = [np.datetime64(start),  np.datetime64(end), "INIT"]
        else:
            print("Pad with epoch")
            if workload == "unet3d":
                df.loc[-1] = [np.datetime64(start),  np.datetime64(end), "EPOCH"]
            else:
                df.loc[-1] = [np.datetime64(start),  np.datetime64(end), "TRAINING"]


    # Plot the events
    for i, _ in enumerate(categories):
        start_dates = mdates.date2num(df.start_date)
        end_dates = mdates.date2num(df.end_date)
        durations = end_dates - start_dates
        xranges = list(zip(start_dates, durations))
        ymin = ymins[i] - 0.5
        yrange = (ymin, bar_height)
        colors = [colors_dict[event] for event in df.event]
        # Plot vertical lines delimiting epoch starts
        if workload == "unet3d":
            ax.vlines(x=start_dates, ymin=ymin, ymax=0.5, color='k', linewidth=0.25)
        ax.broken_barh(xranges, yrange, facecolors=colors, alpha=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Plot vertical lines 
    # vlines format is a dictionary of label -> time
    if vlines:
        for label, xpos in vlines.items():
            if start is not None and end is not None:
                if xpos > start and xpos < end:
                    ax.axvline(x=xpos, linewidth=0.7)
                    ax.annotate(text=label, xy=(xpos, 0), xytext =(xpos,0.8), fontsize=8, rotation=45)
            else:
                ax.axvline(x=xpos, linewidth=0.7)
                ax.annotate(text=label, xy=(xpos, 0), xytext =(xpos,0.8), fontsize=8, rotation=45)

    # Add the legend
    patches = [mpatches.Patch(color=color, label=key) for (key, color) in colors_dict.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left", fontsize=fontsize)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=100))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(xformat))

    # Format the y-ticks
    ax.get_yaxis().set_visible(False)
    ax.set_title("Timeline", fontsize=fontsize+2)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.2, color="grey")
    ax.tick_params(axis="x", which="both", direction="out", rotation=30, labelsize=fontsize)


def plot_pids_timeline_cpu_gpu(data_dir, timeline_dir, workload, title, paper_version=False, name=None, start=None, end=None, xformat="%H:%M", margin=np.timedelta64(1, "s"), filename=None, vlines=None):

    print(f"Generating plot {title}")

    config = os.path.basename(timeline_dir)
    plotting_info = _get_plotting_info_json(timeline_dir)
    timeline_files = plotting_info.keys()

    if filename is None:
        filename = config
        if name is not None:
            filename += '_' + name

    plot_iostat = False
    if (not paper_version) and isfile(join(data_dir, 'iostat.csv')):
        plot_iostat = True

    # Configure plot size, aspect ratios, etc.
    fontsize = 16
    extra_height = 4 if len(timeline_files) == 1 else 1

    if paper_version:
        total_rows = len(timeline_files) + 2
        gridspec_kw={"height_ratios": [1.5] * (total_rows - 1) + [0.5]}
        figsize = (30, (total_rows -1) * 1.5 + extra_height)
    else:
        # Check out how many rows we'll need
        if plot_iostat:
            total_rows = len(timeline_files) + 4
            gridspec_kw={"height_ratios": [1.5] * 2 + [1.5] * 1 + [2.5] * len(timeline_files) + [1]}
        else:
            total_rows = len(timeline_files) + 3
            gridspec_kw={"height_ratios": [2.5] * (total_rows - 1) + [1]}

        figsize = (30, (total_rows -1) * 3 + extra_height)

    # Configure appearance of bpftrace timeline
    trace_plot_config = {
        'bar_height': 1,
        'ymins': [0, 1, 2],
        'categories': ["BIO", "R/W", "OPEN"],
        'colors_dict': dict(
            OPENAT="purple",
            READ="dodgerblue",
            WRITE="red",
            BIOR="blue",
            BIOW="red",
        )
    }
    # Configure appearance of mllog event timeline
    mllog_event_plot_config = {
        'categories': ['Timeline'],
        'ymins': [0],
        'bar_height': 1
    }
    # TODO DLIO
    if workload == "unet3d":
        mllog_event_plot_config['colors_dict'] = dict(INIT="blue", EPOCH="gold", EVAL="darkorchid", CHECKPOINT="mediumvioletred")
    elif workload == "dlrm":
        mllog_event_plot_config['colors_dict'] = dict(INIT="blue", TRAINING="gold", EVAL="darkorchid")
    else:
        mllog_event_plot_config['colors_dict'] = dict(INIT="blue", TRAINING="gold", CHECKPOINT="mediumvioletred")


    fig, axs = plt.subplots(
        nrows=total_rows,
        ncols=1,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        sharex=True,
    )

    # Keep an index over the axes
    i_ax = -1

    if not paper_version:
        i_ax += 1
        plot_cpu_info(data_dir, axs[i_ax], fontsize=fontsize, start=start, end=end)

    i_ax += 1
    plot_gpu_info(data_dir, axs[i_ax], fontsize=fontsize, start=start, end=end)

    if plot_iostat:
        i_ax += 1
        plot_iostat_info(data_dir, axs[i_ax], fontsize=fontsize, start=start, end=end)
    
    i_ax += 1
    for i, timeline_file in enumerate(timeline_files):
        plot_legend = False

        # Only plot the legend for the middle timeline plot
        if (len(timeline_files) > 1 and i == len(timeline_files) // 2) or (len(timeline_files) == 1 and i == 0):
            plot_legend = True

        plot_trace_timeline(timeline_dir, timeline_file, plotting_info, axs[i_ax + i], trace_plot_config, plot_legend, fontsize=fontsize, start=start, end=end, margin=margin)

    # Should be the last axis
    i_ax += 1
    plot_mllog_events(data_dir, axs[i_ax], mllog_event_plot_config, name=name, fontsize=fontsize, start=start, end=end, vlines=vlines)

   

    if not paper_version:
        fig.suptitle(title)
        output_dir = join(data_dir, 'plots', config)
    else:
        output_dir = join(data_dir, 'plots_paper')
    
    # Create output directory if it doesn't exist
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not re.match(r'\.png', filename):
        filename += '.png'

    filename = join(output_dir, filename)
    print(f"Saving figure to {filename}\n")

    plt.tight_layout(pad=0.5, h_pad=0.5)
    plt.savefig(filename, format="png", dpi=500)


def get_plotting_ranges(data_dir, workload):

    df = pd.read_csv(os.path.join(data_dir, "timeline.csv"), names=["start_date", "end_date", "event"])

    init = df.iloc[0]

    if workload == "unet3d":
        first_epoch = df[df["event"] == "EPOCH"].iloc[0]
        first_training = None
    else:
        # For DLRM or BERT, we train on a single epoch, so we use 'TRAINING' events instead
        first_epoch = None
        first_training = df[df["event"] == "TRAINING"].iloc[0]

    # Can add more workload or trace specific logic here to
    # save points of interest before returning plot ranges
    try:
        first_eval = df[df["event"] == "EVAL"].iloc[0]
        second_eval = df[df["event"] == "EVAL"].iloc[1]
        third_eval = df[df["event"] == "EVAL"].iloc[2]
    except:
        print("no eval in this workload")
        first_eval = None
        second_eval = None
        third_eval = None

    last_event = df.iloc[-1]

    # timedeltas to use to extend plotting range slightly 
    td_5s = np.timedelta64(5, 's')
    td_30s = np.timedelta64(30, 's')
    td_2min = np.timedelta64(2, 'm')
    td_5min = np.timedelta64(5, 'm')

    interesting_time_ranges = {
        # "init": (np.datetime64("2022-09-29T19:32:25.00"), np.datetime64(init.end_date) + td_5s),
        # "day_6_file_read": (np.datetime64("2022-09-29T19:32:30.239506016") - td_5s, np.datetime64("2022-09-29T19:32:44.958460081") + td_5s),
        # "day_0_file_read": (np.datetime64("2022-09-29T19:33:01.420377093") - td_5s, np.datetime64("2022-09-29T19:33:15.946566115") + td_5s),
        # "day_1_file_read": (np.datetime64("2022-09-29T19:42:48.089316076") - td_5s, np.datetime64("2022-09-29T19:42:57.752291574") + td_5s),
        # "day_2_file_read": (np.datetime64("2022-09-29T19:56:33.207792720") - td_5s, np.datetime64("2022-09-29T19:56:47.943868838") + td_5s),
        # "day_3_file_read": (np.datetime64("2022-09-29T20:10:30.240881200") - td_5s, np.datetime64("2022-09-29T20:10:39.802377009") + td_5s),
        # "day_4_file_read": (np.datetime64("2022-09-29T20:24:18.938153014") - td_5s, np.datetime64("2022-09-29T20:24:28.575144348") + td_5s),
        # "day_5_file_read": (np.datetime64("2022-09-29T20:33:37.799697746") - td_5s, np.datetime64("2022-09-29T20:33:48.829317627") + td_5s),
        # "Overview": (np.datetime64(init.start_date) - td_5s, np.datetime64(second_eval.end_date) + td_5s) if second_eval is not None else None, 
        "init": (np.datetime64(init.start_date) - td_5s, np.datetime64(init.end_date) + td_5s),
        "first_5min": (np.datetime64(init.start_date) - td_5s, np.datetime64(init.start_date) + td_5min),
        "first_training": (np.datetime64(first_training.start_date) - td_5s, np.datetime64(first_training.end_date) + td_5s) if first_training is not None else None, 
        "first_epoch": (np.datetime64(first_epoch.start_date) - td_5s, np.datetime64(first_epoch.end_date) + td_5s) if first_epoch is not None else None, 
        "first_eval": (np.datetime64(first_eval.start_date) - td_5s, np.datetime64(first_eval.end_date) + td_5s) if first_eval is not None else None,
        "last_2min": (np.datetime64(last_event.end_date) - td_2min, None),
        "last_5s": (np.datetime64(last_event.end_date) - td_5s, None),
    }

    return interesting_time_ranges


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Create the timeline plots for a given run")
    p.add_argument("data_dir", help="Path to 'timeline' subdirectory in preprocessed data directory")
    p.add_argument("workload", help="Workload name", choices=['unet3d', 'bert', 'dlrm', 'dlio'])
    p.add_argument("experiment_name", help="Plot title")
    p.add_argument("-a", "--all-plots", action="store_true", default=False, help="Generate all the default zooms into the timeline (first 5min, first epoch, etc.)")
    args = p.parse_args()

    data_dir = args.data_dir
    workload = args.workload
    title = args.experiment_name
    all_plots = args.all_plots

    if not os.path.isdir(data_dir):
        print(f"ERROR: Invalid data directory")
        exit(-1)

    if not _verify_all_necessary_data_present(data_dir):
        exit(-1)
    
    print(f'All necessary data present')
    plot_all_configs(data_dir, workload, title, all_plots)


    exit()

    # Can mark points of interest with vertical lines by passing
    # timestamps to the plotting function as the vlines argument

    # file_opens = {
    #     "day_6": np.datetime64("2022-09-29T19:32:30.238828817"),    # test data file
    #     "day_0": np.datetime64("2022-09-29T19:33:01.419774134"),
    #     "day_1": np.datetime64("2022-09-29T19:42:48.088704926"),
    #     "day_2": np.datetime64("2022-09-29T19:56:33.207291549"),
    #     "day_3": np.datetime64("2022-09-29T20:10:30.240392129"),
    #     "day_4": np.datetime64("2022-09-29T20:24:18.937630143"),
    #     "day_5": np.datetime64("2022-09-29T20:33:37.386817568"),
    # }




