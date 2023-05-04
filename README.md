## Generating the timeline and instrumentation plots from the paper

Download the timeline and instrumentation data here:

https://mcgill-my.sharepoint.com/:u:/g/personal/oana_balmau_mcgill_ca/EV8SApq87yJFtZfb_n_vgwYBoSZwNaKa4z7lCuEn_EGUmA?e=b8h4jL

https://mcgill-my.sharepoint.com/:u:/g/personal/oana_balmau_mcgill_ca/Ecmosl4mMFlDry-HH8oDDDIBKb1ZQJhLQ4s80M-BSa34JQ?e=HeLksK

Decompress the archives and run
```
./generate_timeline_plots.sh
./generate_instrumentation_plots.sh
```

## Generating new timeline plots

### Preprocessing the raw data

```
$ python3 preprocess_traces.py -h
usage: preprocess_traces.py [-h] [-o OUTPUT_DIR] [-s SKIP_TO] [-ml] traces_dir {unet3d,bert,dlrm,dlio}

Preprocess the output of the tracing scripts for plotting

positional arguments:
  traces_dir            Directory where raw traces are
  {unet3d,bert,dlrm,dlio}
                        Which workload was run

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Processed traces directory. Default will be in a 'processed/' folder in the raw traces directory.
  -s SKIP_TO, --skip-to SKIP_TO
                        Skip to a certain step in the pipeline.
  -ml, --mllog-only     Process only the mllog/dliolog
```

### Plotting the preprocessed data

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

## Generating new instrumentation plots

In this section, we assume you have trace data for a set of runs with the following structure:
```
experiment_name/
    config_1gpu_1batch
    config_1gpu_2batch
    config_1gpu_3batch
    config_2gpu_1batch
    config_2gpu_2batch
    ...
```
Refer to `generate_instrumentation_plots.sh` for an example.

The processing differs based on the workload.

### For UNET3D and DLRM:
  
- Pre-process the raw application log for each configuration and turn it to valid JSON. You can run `preprocess_traces.py` with the `-ml` option to do this (or without the option to process all of the trace data, but we only need the log).
- Create a new directory with all the application logs renamed to the configuration details contained under `raw_data/`:
  ```
  instrumentation_data/
      experiment_name/
          raw_data/
              config_1gpu_1batch.json
              config_1gpu_2batch.json
              config_1gpu_3batch.json
              ...
  ```
  You can use the `prepare_instru_unet_dlrm.sh` script for this.

- Run `proc_instru_data.py` :
  ```
  $ python3 proc_instru_data.py -h
    usage: proc_instru_data.py [-h] [-o OUTPUT_DIR] [-t TITLE] [-l] [-f] [-pb] [-pt] [-pl] [-bh] data_dirs [data_dirs ...] {unet3d,dlrm}

    Generate step breakdowns, throughputs and latencies plots from UNET3D and DLRM instrumentation data.

    positional arguments:
      data_dirs             Data directories
      {unet3d,dlrm}         Workload

    options:
      -h, --help            show this help message and exit
      -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                            Output directory. Defaults to the data directory for single dir and 'data_step_breakdown' for multiple data directories.
      -t TITLE, --title TITLE
                            Additonal string to put after workload name for plots
      -l, --legend          Add legend to plots
      -f, --fit             Fit model to distributions or not
      -pb, --breakdown      Plot the step breakdown.
      -pt, --throughputs    Plot the throughputs.
      -pl, --latencies      Plot the latencies.
      -bh, --big-histo      Save file with all compute time distributions and fits for the annex.
    ```


### For BERT:
For BERT we use Tensorflow profiler traces, and we assume each `experiment_name/config_1gpu_1batch` directory contains multiple `timeline-xxx.json` profiler traces. Simply run `proc_instru_data_bert.py` on the trace directory.
```
$ python3 proc_instru_data_bert.py -h
usage: proc_instru_data_bert.py [-h] [-o OUTPUT_DIR] [-t TITLE] [-b] [-tp] [-bh] [-dp] data_dirs [data_dirs ...]

Plot step breakdown and throughputs from BERT Profiler traces.

positional arguments:
  data_dirs             Data directories

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        (optional) Output directory.
  -t TITLE, --title TITLE
                        (optional) Plot title.
  -b, --breakdown       Plot the step breakdown.
  -tp, --throughputs    Plot the throughputs.
  -bh, --big-histo      Save file with all compute time distributions and fits for the annex.
  -dp, --do-processing  Whether to proces raw data (long) or use saved file
```

### For DLIO (Benchmark):
Similarly, you only need the `dlio.log` files for each configuration. Use the `proc_instru_data_dlio.py` script.
```
$ python3 proc_instru_data_dlio.py -h
usage: Process DLIO instrumentation data and generate plots [-h] [-o OUTPUT_DIR] [-t TITLE] [-l] [-pb] [-pt] [-pl] data_dirs [data_dirs ...] {unet3d,bert,dlrm}

positional arguments:
  data_dirs             Data directories
  {unet3d,bert,dlrm}    Workload

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory
  -t TITLE, --title TITLE
                        Title for plots
  -l, --legend          Add the legend
  -pb, --breakdown      Plot the step breakdown.
  -pt, --throughputs    Plot the throughputs.
  -pl, --latencies      Plot the latencies.
``` 