import os
import numpy as np

from file_processing import mp3_to_raw_data, raw_data_to_wav

def get_raw_data(sample_rate):
    base_path = '../fma_small/000/'
    dir_files = os.listdir(base_path)
    data_list = []
    for filename in dir_files:
        file_path = base_path+filename
        print(file_path)
        file_data = mp3_to_raw_data(file_path,sample_rate)
        data_list.append(file_data)

    return np.concatenate(data_list,axis=0)
