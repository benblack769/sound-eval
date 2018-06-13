import os
import numpy as np
import itertools

from file_processing import mp3_to_raw_data, raw_data_to_wav

def get_subfolder_data(base_path, sample_rate):
    dir_files = os.listdir(base_path)
    data_list = []
    for filename in dir_files:
        file_path = base_path+filename
        print(file_path)
        file_data = mp3_to_raw_data(file_path,sample_rate)
        data_list.append(file_data)

    return data_list

def get_raw_data(sample_rate, num_folders=1):
    return np.concatenate(get_raw_data_list(sample_rate,num_folders)[1])

def get_raw_data_list(sample_rate, num_folders=1):
    base_paths = ['../fma_small/00{}/'.format(i) for i in range(num_folders)]
    subfolder_data = sum([get_subfolder_data(fold,sample_rate) for fold in base_paths],[])
    return base_paths,subfolder_data
