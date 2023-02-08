import os
import re
import pathlib

import numpy as np

from pprint import PrettyPrinter

from preproc.utilities import get_fields



def get_pid_file_mapping(parent_pids: set, dataloader_pids: set, config, outdir) -> dict:
    """
    Returns a map of pid -> file used to separate the traces into individual timelines for plotting.
    """
    if config == "all_combined":
        # Map the same single file to each pid in parents U dataloaders
        return {pid : os.path.join(outdir, 'combined.log') for pid in parent_pids.union(dataloader_pids)}
    elif config == "each_parent":
        # In this config, we plot each parent individually
        # and merge with their dataloaders (if present)
        num_parents = len(parent_pids)
        num_loaders = len(dataloader_pids)

        # We will sort the PIDs in this case
        # Since they are created by the OS in increasing order,
        # we will assume each parent creates a loader one by one (in reality it's in parallel)
        # and associate loaders to parents in round-robin on the sorted list.
        # Even if the order will not be exactly the truth, we should have one loader per epoch
        # per parent, which is OK. 
        parent_pids = list(parent_pids)
        dataloader_pids = list(dataloader_pids)
        parent_pids.sort()
        dataloader_pids.sort()

        mapping = {}
        files = []
        for i, pid in enumerate(parent_pids):
            file = os.path.join(outdir, f'parent_{i}.log')
            mapping[pid] = file
            files.append(file)

        if len(dataloader_pids) > 0:
            if num_loaders % num_parents != 0:
                raise Exception(f'ERROR: dataloaders cannot be distributed evenly between parents: {num_parents} vs {num_loaders}')

            for i, loader_pid in enumerate(dataloader_pids):
                file_idx = i % num_parents
                print(f'Associating loader pid {loader_pid} to file {file_idx}')
                mapping[pid] = files[file_idx]

        return mapping
    elif config == "parents_combined_loaders_combined":

        # In this config we have 2 files, 1 for all parents combined, another for all loaders combined
        # This config is only presesent if loaders are present.
        mapping = {pid : os.path.join(outdir, 'parent_0.log') for pid in parent_pids}
        for pid in dataloader_pids:
            mapping[pid] = os.path.join(outdir, 'loader_0.log')
        return mapping

    elif config == "each_parent_and_their_loaders":
        # Only present if len(dataloader_pids) > 0
        #
        # In this config, we will have a separate file for each parent and for each set
        # of parent-grouped loaders. 
        # I.e. if we have parent pids {1,2} and loaders {3,4,5,6}
        # we will have the folowing mapping of pid to file:
        # 1: parent_1
        # 2: parent_2 
        # 3, 5: loaders_1
        # 4, 6: loaders_2
        mapping = {}
        num_parents = len(parent_pids)
        num_loaders = len(dataloader_pids)

        # We will sort the PIDs in this case
        # Since they are created by the OS in increasing order,
        # we will assume each parent creates a loader one by one (in reality it's in parallel)
        # and associate loaders to parents in round-robin on the sorted list.
        # Even if the order will not be exactly the truth, we should have one loader per epoch
        # per parent, which is OK. 
        parent_pids = list(parent_pids)
        dataloader_pids = list(dataloader_pids)
        parent_pids.sort()
        dataloader_pids.sort()

        for i, pid in enumerate(parent_pids):
            mapping[pid] = os.path.join(outdir, f'parent_{i}.log')

        for i, pid in enumerate(dataloader_pids):
            mapping[pid] = os.path.join(outdir, f'loader_{i % num_parents}.log')

        return mapping
    else:
        raise Exception("Uknown PID splitting configuration!")



def prepare_traces_for_timeline_plot(traces_dir, parent_pids: set, dataloader_pids: set, ignore_pids: set, TRACES, TRACE_LATENCY_COLUMN_IDX):

    # Will create three subdrectories with different representations on the timeline
    # Different possibilities:
    # - one file per parent, one file for each parent's associated dataloaders
    # - one file per (parent + associated dataloaders)
    # - all parents combined, all dataloaders combined
    # - all combined

    plot_configs = ["all_combined", "each_parent"]

    if len(dataloader_pids) > 0:
        plot_configs.extend(["parents_combined_loaders_combined", "each_parent_and_their_loaders"])

    ignore = False
    if len(ignore_pids) > 0:
        ignore = True
        # Create a regex from the ignore pid list
        # of the form 'pid1|pid2|pid3|...'
        p_ignore_pids = "|".join([str(pid) for pid in ignore_pids])
        p_ignore_pids = re.compile(rf'.*{p_ignore_pids}.*')

    # At this point, all traces have the same first 2 fields: UTC timestamp and PID
    # e.g. 2023-01-27T18:56:05.163666652 1702533
    p_data = re.compile(r'^\d{4}\-\d{2}\-\d{2}T\d{2}\:\d{2}\:\d{2}\.\d{9}\s+(\d+)')

    pp = PrettyPrinter(indent=4)

    for plot_config in plot_configs:
        # Create subdirectories under timeline/pid for each configuration
        outdir = os.path.join(traces_dir, 'timeline', 'pid', plot_config)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

        mapping = get_pid_file_mapping(parent_pids, dataloader_pids, plot_config, outdir)
        pp.pprint(mapping)

        # Create shared lists to hold all the data
        # mapping.values() has multiple instances of the same files
        shared_lists = {}
        for file in mapping.values():
            if file not in shared_lists:
                shared_lists[file] = []
                
        for trace in TRACES:
            print(f"Preparing trace {trace} for timeline plot")

            tracefile = os.path.join(traces_dir, f'{trace}_UTC.out')

            with open(tracefile, 'r') as tracefile:

                for line in tracefile:
                    # Skip unwanted PID lines
                    if ignore and re.match(p_ignore_pids, line):
                        continue
                
                    match = re.match(p_data, line)
                    pid = match.group(1)
                    
                    # We may still get some unknown PIDs here, just ignore them
                    if pid in mapping:

                        latency_idx = TRACE_LATENCY_COLUMN_IDX[trace]
                        data = get_fields(line)
                        if trace == 'bio':
                            lat = int(data[latency_idx])
                            end_time = np.datetime64(data[0])
                            bio_type = 'BIOR' if data[4] == 'R' else 'BIOW'
                            line = f'{end_time - lat},{end_time},{bio_type}\n'
                        elif trace == 'read':
                            lat = int(data[latency_idx])
                            end_time = np.datetime64(data[0])
                            line = f'{end_time - lat},{end_time},READ\n'

                        elif trace == 'write':
                            lat = int(data[latency_idx])
                            end_time = np.datetime64(data[0])
                            line = f'{end_time - lat},{end_time},WRITE\n'
                        
                        elif trace == 'openat':
                            lat = int(data[latency_idx])
                            end_time = np.datetime64(data[0])
                            line = f'{end_time - lat},{end_time},OPENAT\n'
                        else:
                            raise Exception(f"Unknown trace type: {trace}")

                        shared_lists[mapping[pid]].append(line)

        for file in shared_lists:
            # At this point, the list contains data for each trace
            # separately. Sort them to mix the different operations
            # THE POWER OF MODERN COMPUTERS
            # Sorting a 20MB list in half a second!
            shared_lists[file].sort()
            if os.path.isfile(file):
                os.remove(file)
            with open(file, 'w') as outfile:
                outfile.writelines(shared_lists[file])
