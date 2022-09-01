import json
import code
import os.path
import pathlib
import argparse
import numpy as np
import pandas as pd
from matplotlib import dates as mdates, pyplot as plt, patches as mpatches, colors


def plot_pids_timeline_cpu_gpu(data_dir, title, start=None, end=None, xformat="%H:%M", margin=np.timedelta64(60, "s"), filename=None):

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

    fig, axs = plt.subplots(
        nrows=len(pids) + 3,
        ncols=1,
        figsize=(30, len(pids) * 3),
        gridspec_kw={"height_ratios": [3] * (len(pids) + 2) + [1]},  # 1 for timeline
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
    ax.set_ylabel("percent utilisation(%)")

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
    df = pd.read_csv(os.path.join(data_dir, "gpu_data/gpu_avg.csv"), sep=",")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if start is not None:
        df = df[df["timestamp"] >= np.datetime64(start)]
    if end is not None:
        df = df[df["timestamp"] <= np.datetime64(end)]

    ax1 = axs[1]
    ax1.set_title("GPU Usage")
    ax1.set_ylabel("percent utilisation(%)")

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
        label="Framebuffer memory use (MB)",
        color="tab:blue",
        linewidth=1.5,
        markersize=5,
    )

    ax1.grid(True, which="both", linestyle="--")
    ax1.tick_params(which="both", direction="in", grid_color="grey", grid_alpha=0.2)

    ax1.set_ylim(ymin=0)
    ax2.set_ylim(ymin=0)

    ax1.legend(loc="center left")
    ax2.legend(loc="center right")


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
        if i == 3:
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
        print("Pad with epoch")
        df.loc[-1] = [np.datetime64(start),  np.datetime64(end), "EPOCH"]

    categories = ["Timeline"]

    ymins = [0]
    colors_dict = dict(INIT="blue", EPOCH="gold", EVAL="darkorchid")

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
        ax.vlines(x=start_dates, ymin=ymin, ymax=0.5, color='k', linewidth=0.25)
        ax.broken_barh(xranges, yrange, facecolors=colors, alpha=0.8)
        # you can set alpha to 0.6 to check if there are some overlaps

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add the legend
    patches = [mpatches.Patch(color=color, label=key) for (key, color) in colors_dict.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc="center left")

    # TODO: Calculate the sort of timescale we're plotting and choose appropriate limits
    # Format the x-ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=100))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(
            xformat,
    ))

    # Format the y-ticks
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    ax.grid(True, axis="x", linestyle="--", linewidth=0.45, alpha=0.2, color="grey")
    ax.tick_params(axis="x", which="both", direction="out", rotation=30)


    fig.suptitle(title)

    if filename is not None:
        filename = filename
    else:
        filename = "timelines/cpu_gpu_timeline"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    pathlib.Path(os.path.join("./plots/", output_dir)).mkdir(parents=True, exist_ok=True)

    print(f"Saving figure to plots/{filename}")
    plt.savefig(f"./plots/{filename}", format="png", dpi=500)



def get_plotting_ranges(data_dir):

    df = pd.read_csv(os.path.join(data_dir, "mllog_data/timeline.csv"), names=["start_date", "end_date", "event"])

    init = df.iloc[0]
    first_epoch = df[df["event"] == "EPOCH"].iloc[0]
    first_eval = df[df["event"] == "EVAL"].iloc[0]
    last_event = df.iloc[-1]

    td_5s = np.timedelta64(5, 's')
    td_30s = np.timedelta64(30, 's')
    td_2min = np.timedelta64(2, 'm')
    td_5min = np.timedelta64(5, 'm')

    interesting_time_ranges = {
        "init": (np.datetime64(init.start_date), np.datetime64(init.end_date)),
        "first_30s": (np.datetime64(init.start_date), np.datetime64(init.start_date) + td_30s),
        "first_5min": (np.datetime64(init.start_date), np.datetime64(init.start_date) + td_5min),
        "first_epoch": (np.datetime64(first_epoch.start_date) - td_5s, np.datetime64(first_epoch.end_date) + td_5s),
        "first_eval": (np.datetime64(first_eval.start_date) - td_5s, np.datetime64(first_eval.end_date) + td_5s),
        "last_2min": (np.datetime64(last_event.end_date) - td_2min, np.datetime64(last_event.end_date)),
        "last_5s": (np.datetime64(last_event.end_date) - td_5s, np.datetime64(last_event.end_date)),
    }

    # code.interact(local=locals())

    return interesting_time_ranges


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Create the timeline plots for a given run")
    p.add_argument("data_dir", help="Directory where the preprocessed traces are")
    p.add_argument("experiment_name", help="Title of the plot")
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Invalid trace directory")
        exit(-1) 

    plot_pids_timeline_cpu_gpu(
        args.data_dir,
        title=args.experiment_name,
        filename=f"timelines/{args.experiment_name}/overview.png",
    )

    # Extract times of first epoch, first eval, first 5 min and last 5 minutes from the mllog file
    interesting_time_ranges = get_plotting_ranges(args.data_dir)

    for name, time_range in interesting_time_ranges.items():

        start = time_range[0]
        end = time_range[1]

        plot_pids_timeline_cpu_gpu(
            args.data_dir,
            title=f"{args.experiment_name} - {name}",
            start=start,
            end=end,
            xformat="%H:%M:%S",
            margin=np.timedelta64(1, "s") if name != "init" else np.timedelta64(100, "ms"),
            filename=f"timelines/{args.experiment_name}/{name}.png",
        )

