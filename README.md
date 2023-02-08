

## Preprocessing the raw data

```
$ python3 preprocess_traces.py -h
usage: preprocess_traces.py [-h] [-o OUTPUT_DIR] traces_dir {unet3d,bert,dlrm,dlio}

Preprocess the output of the tracing scripts for plotting

positional arguments:
  traces_dir            Directory where raw traces are
  {unet3d,bert,dlrm,dlio}
                        Which workload was run

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Processed traces directory. Default is 'data_processed/'
```

## Plotting the preprocessed data

Plots will be created under `plots/` in the preprocessed data directory.
```
$ python3 plot_timelines.py -h
usage: plot_timelines.py [-h] [-a] data_dir {unet3d,bert,dlrm,dlio} experiment_name

Create the timeline plots for a given run

positional arguments:
  data_dir              Path to 'timeline' subdirectory in preprocessed data directory
  {unet3d,bert,dlrm,dlio}
                        Workload name
  experiment_name       Plot title

optional arguments:
  -h, --help            show this help message and exit
  -a, --all-plots       Generate all the default zooms into the timeline (first 5min, first epoch, etc.)
```
