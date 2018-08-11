import numpy as np
import argparse
import os
import process_many_files

def load_spec_list(base_folder):
    all_filenames = process_many_files.get_all_paths(base_folder,"npy")
    all_file_datas = [np.load(os.path.join(base_folder,fname)) for fname in all_filenames]
    return all_filenames,all_file_datas

def make_dirs(paths):
    for path in paths:
        pathdir = os.path.dirname(path)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

def sum_single(src_data, sum_over):
    drop_len = (len(src_data)//sum_over)*sum_over
    dropped_data = src_data[:drop_len]
    reshaped_data = np.reshape(dropped_data,(drop_len//sum_over,sum_over,src_data.shape[1]))
    summed_data = 

def calc_all_vectors(source_dir, dest_dir, sum_over):
    all_filenames = process_many_files.get_all_paths(source_dir,"npy")
    source_abs_filenames = [os.path.join(source_dir,filename) for filename in all_filenames]
    dest_abs_filenames = [os.path.join(dest_dir,filename) for filename in all_filenames]

    make_dirs(dest_abs_filenames)
    for src_path,dest_path in zip(source_abs_filenames,dest_abs_filenames):
        src_data = np.load(src_path)
        dest_data = sum_single(src_data, sum_over)
        np.save(dest_path,dest_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into a folder of .npy vector files")
    parser.add_argument('vector_dataset', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('--sum_over', target="sum_amnt" help='Path to learning model folder.')
    parser.add_argument('output_folder', help='Path to output folder where files will be stored.')

    args = parser.parse_args()

    calc_all_vectors(args.vector_dataset,args.output_folder,args.sum_amnt)
