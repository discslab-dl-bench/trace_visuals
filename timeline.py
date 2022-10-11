import json
import os.path
import pathlib
import argparse
import numpy as np
import pandas as pd
from matplotlib import dates as mdates, pyplot as plt, patches as mpatches, colors
from pyrsistent import v


def plot_pids_timeline_cpu_gpu(data_dir, workload, title, long=True, name=None, start=None, end=None, xformat="%H:%M", margin=np.timedelta64(5, "s"), filename=None, vlines=None):

    print(f"Generating plot {title}")

    pid_names_file = os.path.join(data_dir, "pids.json")

    if not os.path.isfile(pid_names_file):
        print(f"ERROR: Missing pids.json file in {data_dir}")
        exit(-1) 

    pid_names = open(pid_names_file, 'r')
    pid_names = json.load(pid_names)
    pids = list(pid_names.keys())


    bar_height = 1
    ymins = [0, 1, 2]
    categories = ["BIO", "R/W", "OPEN"]
    colors_dict = dict(
        OPENAT="purple",
        READ="dodgerblue",
        WRITE="red",
        BIOR="blue",
        BIOW="red",
    )

    extra_height = 4 if len(pids) == 1 else 1

    if long:
        figsize = (30, len(pids) * 3 + extra_height)
        gridspec_kw={"height_ratios": [2.5] * (len(pids) + 2) + [1]}
    else:
        figsize = (20, len(pids) * 3 + extra_height)
        gridspec_kw={"height_ratios": [2.5] * (len(pids) + 2) + [1]}

    fig, axs = plt.subplots(
        nrows=len(pids) + 3,
        ncols=1,
        figsize=figsize,
        gridspec_kw=gridspec_kw, # 1 for timeline
        sharex=True,
    )

    #
    # Plot CPU
    #
    df = pd.read_csv(
        os.path.join(data_dir, "cpu_data/cpu_all.csv"),
        sep=",",
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= np.datetime64(start)]
    if end is not None:
        df = df[df["timestamp"] <= np.datetime64(end)]

    ax = axs[0]
    ax.set_title("CPU Usage")
    ax.set_ylabel("Percent Use (%)")

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
        line = ax.plot(df["timestamp"], df[var], label=var, linewidth=1)
        line[0].set_color(cm(1 * i / n_features))

    ax.grid(True, which="both", linestyle="--", color="grey", alpha=0.2)
    ax.tick_params(which="both", direction="in")

    ax.set_ylim(ymin=0)
    ax.legend(bbox_to_anchor=(1, 0.5), loc="center left")

    #
    # Plot GPU
    #
    df = pd.read_csv(os.path.join(data_dir, "gpu_data/gpu_avg.csv"), sep=",", on_bad_lines='skip') # add additional argument on_bad_lines='skip' to plot
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if start is not None:
        df = df[df["timestamp"] >= np.datetime64(start)]
    if end is not None:
        df = df[df["timestamp"] <= np.datetime64(end)]

    ax1 = axs[1]
    ax1.set_title("GPU Usage")
    ax1.set_ylabel("Percent Use (%)")

    ax1.plot(
        df["timestamp"],
        df["sm"],
        label="GPU MultiProcessor Use (%)",
        color="tab:red",
        linewidth=1,
        markersize=5,
    )
    ax1.plot(
        df["timestamp"],
        df["mem"],
        label="GPU Memory Use (%)",
        color="tab:orange",
        linewidth=1,
        markersize=5,
    )

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Size (MB)")
    ax2.plot(
        df["timestamp"],
        df["fb"],
        label="Framebuffer Memory Use (MB)",
        color="tab:blue",
        linewidth=1.5,
        markersize=5,
        rasterized=True
    )

    ax1.grid(True, which="both", linestyle="--")
    ax1.tick_params(which="both", direction="in", grid_color="grey", grid_alpha=0.2)

    ax1.set_ylim(ymin=0, ymax=100)
    ax2.set_ylim(ymin=0)

    # This will combine the GPU %mp, %mem and FBmem legends  
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    ax2.legend(handles, labels, loc='center left')

    # Plot PIDs
    #
    for i, pid in enumerate(pids):
        print(f"Processing pid {pid}")

        df = pd.read_csv(
            os.path.join(data_dir, f"st_end_data/st_end_data_{pid}"), names=["start_date", "end_date", "event"]
        )
        df = df[["start_date", "end_date", "event"]]
        df.start_date = pd.to_datetime(df.start_date).astype(np.datetime64)
        df.end_date = pd.to_datetime(df.end_date).astype(np.datetime64)
        if start is not None:
            df = df[df["start_date"] >= np.datetime64(start)]
        if end is not None:
            df = df[df["end_date"] <= np.datetime64(end)]

        # If the DataFrame is empty after filtering, skip
        if len(df) == 0:
            print(f"This timerange is empty for pid {pid}. Skipping.")
            continue

        # Can't define this earlier
        masks = {
            "BIO": (df["event"] == "BIOR") | (df["event"] == "BIOW"),
            "OPEN": (df["event"] == "OPENAT") ,
            "R/W": (df["event"] == "READ") | (df["event"] == "WRITE"),
        }

        ax = axs[i + 2]
        if pid in pid_names:
            ptitle = pid_names[pid] 
        else:
            ptitle = pid

        ax.set_title(f"{ptitle}")

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
        ax.tick_params(which="both", direction="in")

        # Format the y-ticks
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)

        # Add the legend
        if (len(pids) > 1 and i == len(pids) // 2) or (len(pids) == 1 and i == 0):
            patches = [
                mpatches.Patch(color=color, label=key) for (key, color) in colors_dict.items()
            ]
            ax.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left")

    # Set the x axis limits
    # We do this here so that we create a margin around the trace data min/max vs. the timeline
    # data which we care less about. This makes the start/end setting work more as expected.
    if margin is None:
        margin = np.timedelta64(5, "s")

    # Sometimes the range we try to plot contains nothing so the limits are NaT
    # and the program throws a value error "Axis limits cannot be NaN or Inf"
    # embed()
    try:
        ax.set_xlim(
            df.start_date.min() - margin,
            df.end_date.max() + margin,
        )
    except Exception as e:
        print(f"Exception caught while trying to set graph limits: {e}")
        print("Skipping this graph.")
        return
    #
    # Plot the timeline
    #
    print(f"Processing timeline")

    df = pd.read_csv(os.path.join(data_dir, "mllog_data/timeline.csv"), names=["start_date", "end_date", "event"])

    df = df[["start_date", "end_date", "event"]]
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
            if workload == "imseg":
                df.loc[-1] = [np.datetime64(start),  np.datetime64(end), "EPOCH"]
            else:
                df.loc[-1] = [np.datetime64(start),  np.datetime64(end), "TRAINING"]


    categories = ["Timeline"]

    ymins = [0]

    # Logical training events for each workload
    if workload == "imseg":
        colors_dict = dict(INIT="blue", EPOCH="gold", EVAL="darkorchid")
    elif workload == "dlrm":
        colors_dict = dict(INIT="blue", TRAINING="gold", EVAL="darkorchid")
    else:   # E.g. BERT
        colors_dict = dict(INIT="blue", TRAINING="gold", CHECKPOINT="mediumvioletred")
    # TODO DLIO?

    # Select the last axes
    ax = axs[-1]

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
        if workload == "imseg":
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
    ax.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left")

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=100))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(xformat))

    # Format the y-ticks
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.2, color="grey")
    ax.tick_params(axis="x", which="both", direction="out", rotation=30)

    # fig.suptitle(title)

    if filename is not None:
        filename = filename
    else:
        filename = "timelines/cpu_gpu_timeline"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    pathlib.Path(os.path.join("./plots/", output_dir)).mkdir(parents=True, exist_ok=True)

    print(f"Saving figure to plots/{filename}\n")
    plt.tight_layout(pad=0.5, h_pad=0.5)
    plt.savefig(f"./plots/{filename}", format="png", dpi=550)



def get_plotting_ranges(data_dir, workload):

    df = pd.read_csv(os.path.join(data_dir, "mllog_data/timeline.csv"), names=["start_date", "end_date", "event"])

    init = df.iloc[0]

    if workload == "imseg":
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
        "last_2min": (np.datetime64(last_event.end_date) - td_2min, np.datetime64(last_event.end_date)),
        "last_5s": (np.datetime64(last_event.end_date) - td_5s, np.datetime64(last_event.end_date)),
    }

    return interesting_time_ranges


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Create the timeline plots for a given run")
    p.add_argument("data_dir", help="Directory where the preprocessed traces are")
    p.add_argument("workload", help="Name of the workload")
    p.add_argument("experiment_name", help="Title of the plot")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Invalid trace directory")
        exit(-1) 

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

    plot_pids_timeline_cpu_gpu(
        args.data_dir,
        args.workload,
        title=args.experiment_name,
        filename=f"timelines/{args.experiment_name}/{args.workload}_overview.png",
        long=True,
    )


    # Extract times of first epoch, first eval, first 5 min and last 5 minutes from the mllog file
    interesting_time_ranges = get_plotting_ranges(args.data_dir, args.workload)

    for name, time_range in interesting_time_ranges.items():
        if time_range is None:
            continue

        plot_pids_timeline_cpu_gpu(
            args.data_dir,
            args.workload,
            title = f"{args.experiment_name} - {name}",
            name = name,
            start = time_range[0],
            end = time_range[1],
            xformat = "%H:%M:%S",
            margin = np.timedelta64(1, "s") if name != "init" else np.timedelta64(100, "ms"),
            filename = f"timelines/{args.experiment_name}/{name}.png",
        )

