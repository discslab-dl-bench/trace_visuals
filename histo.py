import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
    
def plot_histogram(data, title='histo', nbins=100):

    fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(15,15))

    data = np.asarray(data)

    ax.hist(data, bins=nbins)

    median = np.median(data)
    trans = ax.get_xaxis_transform()
    ax.axvline(median, color='k', linestyle='dashed', linewidth=1)
    plt.text(median * 1.5, .85, f'median: {int(median):,}', transform=trans)

    # Create output directory if it doesn't exist
    outdir = Path('histograms')
    outdir.mkdir(parents=True, exist_ok=True)

    figure_filename = outdir / f'{title}.png'

    plt.savefig(figure_filename, format="png", dpi=250)
    # Clear the current axes.
    plt.cla() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)



if __name__ == '__main__':

    data =  [7347,  26701,  2112,  2097216,  2112,  524352,  1088,  1088,  19960448,  8846464,  3798144,  524416,  10364032,  1664,  3645568,  790144,  32384,  5248,  1130624,  1152,  6108800,  79488,  2176,  499840,  7296,  6640256,  131200,  55424,  18560,  1962752,  4224,  4194432,  4224,  3189,  98]

    plot_histogram(data)
