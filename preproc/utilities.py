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

    
    