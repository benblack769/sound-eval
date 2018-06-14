import os
import numpy as np
import itertools
import subprocess

from file_processing import mp3_to_raw_data, raw_data_to_wav

def get_all_music_paths():
    find_call = "find ../fma_small -type f"
    file_list = subprocess.check_output(find_call.split(" ")).decode('utf-8').split()
    mp3_list = [name for name in file_list if name[-4:] == ".mp3"]
    return mp3_list

def get_raw_data(sample_rate, num_files=20):
    return np.concatenate(get_raw_data_list(sample_rate,num_folders)[1])

def get_raw_data_list(sample_rate, num_files=20):
    all_base_paths = get_all_music_paths()
    if len(all_base_paths)  < num_files:
        print("Warning: Querrired too many files")

    subfolder_data = []
    base_paths = []
    for file_path_idx in range(min(num_files, len(all_base_paths))):
        file_path = all_base_paths[file_path_idx]
        try:
            subfolder_data.append(mp3_to_raw_data(file_path,sample_rate))
            base_paths.append(file_path)
        except subprocess.CalledProcessError:
            with open("log/failed_file_loads.txt",'a') as logfile:
                logfile.write("process error on {} with sample rate {}\n".format(file_path,sample_rate))

    return base_paths,subfolder_data
