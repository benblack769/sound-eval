import os
import numpy as np
import itertools
import subprocess
import concurrent.futures

from file_processing import mp3_to_raw_data, raw_data_to_wav

def get_all_music_paths():
    find_call = "find ../fma_small -type f"
    file_list = subprocess.check_output(find_call.split(" ")).decode('utf-8').split()
    mp3_list = [name for name in file_list if name[-4:] == ".mp3"]
    return mp3_list

def get_raw_data(sample_rate, num_files=20):
    return np.concatenate(get_raw_data_list(sample_rate,num_folders)[1])

def mp3_caller(path_rate_pair):
    return mp3_to_raw_data(path_rate_pair[0],path_rate_pair[1])

def process_files_parallel(full_base_paths,sample_rate):
    paths_with_rate = [(path,sample_rate) for path in full_base_paths]
    with concurrent.futures.ProcessPoolExecutor() as pool:#try exchanginhg with ThreadPoolExecutor for additional speed
        return pool.map(mp3_caller,paths_with_rate)

def process_files_sequential(full_base_paths,sample_rate):
    return  [mp3_to_raw_data(file_path,sample_rate) for file_path in full_base_paths]

def get_raw_data_list(sample_rate, num_files=20):
    all_base_paths = get_all_music_paths()
    if len(all_base_paths)  < num_files:
        print("Warning: Querrired too many files")

    full_base_paths = all_base_paths[:min(num_files, len(all_base_paths))]

    full_subfolder_data = process_files_parallel(full_base_paths,sample_rate)

    subfolder_data = [data for data in full_subfolder_data if data is not None]
    base_paths = [path for data,path in zip(full_subfolder_data,full_base_paths) if data is not None]
    return base_paths,subfolder_data
