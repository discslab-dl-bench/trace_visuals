import re
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt


def get_fields(line):
    """
        Split the line on whitespace, join it on a single space then split it again.
        This makes it return nicely delimited tokens because the original number of
        spaces is variable.
    """
    return " ".join(line.split()).split(" ")
    
    
def plot_histogram(data, outdir, title, filename, nbins=100):

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(15,15))
    fig.suptitle(title)

    data = np.asarray(data)

    ax.hist(data, bins=nbins)

    median = np.median(data)
    trans = ax.get_xaxis_transform()
    ax.axvline(median, color='k', linestyle='dashed', linewidth=1)
    plt.text(median * 1.5, .85, f'median: {int(median):,}', transform=trans)

    # Create output directory if it doesn't exist
    pathlib.Path(os.path.join(outdir, "histograms")).mkdir(parents=True, exist_ok=True)

    filename += '.png'
    figure_filename = os.path.join(outdir, "histograms", filename)

    plt.savefig(figure_filename, format="png", dpi=250)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)


def get_local_date(gpu_trace) -> str:
    """
    Return the local date from the nvidia-smi pmon trace in YYYYMMDD format.
    """
    pat = re.compile(r'^\s+(\d{8})\s+(\d{2}:\d{2}:\d{2}).*')

    with open(gpu_trace, 'r') as gpu_trace:
        localdate = None
        for line in gpu_trace:
            if match := pat.match(line):
                localdate = match.group(1)
                print(f"Found the local date in gpu trace: {localdate}\n")
                break
    return localdate


def get_num_gpus(gpu_trace) -> int:
    """
    Return the number of GPUs used in the experiment.
    We assume GPUs are handed out starting from idx 0, which is the case for our workloads.
    """
    # Line format is 
    # 
    # 20230118   09:47:05      4    2647252     C     0     0     -     -   308   python 
    # or  
    # 20230118   09:46:56      5          -     -     -     -     -     -     -   -     
    #
    #  This will match only those lines with a process on GPU and have the GPU index in group 1
    pat = re.compile(r'^\s+\d{8}\s+\d{2}:\d{2}:\d{2}\s+(\d+)\s+(\d+)')

    highest_used_gpu_idx = 0

    with open(gpu_trace, 'r') as gpu_trace:
        for line in gpu_trace:
            if match := pat.match(line):
                highest_used_gpu_idx = max(match.group(1), highest_used_gpu_idx)

    return highest_used_gpu_idx + 1

    
    