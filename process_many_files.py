import os
import numpy as np
import itertools
import subprocess
import concurrent.futures

from file_processing import mp3_to_raw_data, raw_data_to_wav

def get_all_music_paths(base_folder):
    find_call = ["find", ".", "-name", "*.mp3", "-type", "f"]
    file_list = subprocess.check_output(find_call,cwd=base_folder).decode('utf-8').strip().split()
    return file_list

def filter_number(item_list, max_num):
    return item_list[:min(len(item_list),max_num)]

def get_raw_data(sample_rate, num_files=20):
    return np.concatenate(get_raw_data_list(sample_rate,num_files)[1])

def mp3_caller(path_rate_pair):
    return mp3_to_raw_data(path_rate_pair[0],path_rate_pair[1])

def process_files_parallel(full_base_paths,sample_rate):
    paths_with_rate = [(path,sample_rate) for path in full_base_paths]
    with concurrent.futures.ProcessPoolExecutor() as pool:#try exchanginhg with ThreadPoolExecutor for additional speed
        return pool.map(mp3_caller,paths_with_rate)

def process_files_sequential(full_base_paths,sample_rate):
    return  [mp3_to_raw_data(file_path,sample_rate) for file_path in full_base_paths]

def get_raw_data_iter(sample_rate, base_folder, max_num_files=20):
    all_base_paths = get_all_music_paths(base_folder)
    #if len(all_base_paths)  < num_files:
    #    print("Warning: Querrired too many files")
    print("start")
    full_base_paths = all_base_paths[:min(max_num_files, len(all_base_paths))]

    rel_paths = [os.path.join(base_folder,path) for path in full_base_paths]
    return process_files_parallel(rel_paths,sample_rate),full_base_paths
