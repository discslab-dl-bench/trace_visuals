import os
import pathlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from itertools import zip_longest


def iostat_trace_is_present(traces_dir):
    iostat_trace = get_iostat_trace(traces_dir)
    return os.path.isfile(iostat_trace)

def strace_is_present(traces_dir):
    strace_trace = get_strace_trace(traces_dir)
    return os.path.isfile(strace_trace)

def get_dlio_log(traces_dir):
    log_path = os.path.join(traces_dir, 'dlio.log')

    if not os.path.isfile(log_path):
        return str(next(Path(traces_dir).rglob('dlio.log')))

    return log_path

def get_iostat_trace(traces_dir):
    return os.path.join(traces_dir, 'iostat.json')

def get_strace_trace(traces_dir):
    return os.path.join(traces_dir, 'strace.out')

def get_time_align_trace(traces_dir):
    return os.path.join(traces_dir, 'trace_time_align.out')

def get_bio_trace(traces_dir):
    return os.path.join(traces_dir, 'bio.out')

def get_read_trace(traces_dir):
    return os.path.join(traces_dir, 'read.out')

def get_write_trace(traces_dir):
    return os.path.join(traces_dir, 'write.out')

def get_gpu_trace(traces_dir):
    return os.path.join(traces_dir, 'gpu.out')

def get_cpu_trace(traces_dir):
    return os.path.join(traces_dir, 'cpu.out')

def _get_canonical_event_name(evt):
    """
    The three workloads don't agree what a training event looks like.
    """
    if evt == 'EPOCH' or evt == 'BLOCK' or evt == 'TRAINING' or evt == 'RUN':
        evt = 'TRAINING'
    return evt
    
def sliding_window(iterable, n, fillvalue=None):
    """
    Creates an array of n parallel iterators, that are 
    called in round-robin fashion by zip_longest.
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


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

    
    