import numpy as np
import argparse
import yaml
import os
import process_many_files
import tensorflow as tf
from linearlizer import Linearlizer
from calc_all_vecs import run_spec_list

def calc_all_vectors(all_filenames, source_dir, model_path, config):
    source_abs_filenames = [os.path.join(source_dir,filename) for filename in all_filenames]

    source_datas = [np.load(path) for path in source_abs_filenames]
    dest_datas = run_spec_list(source_datas,config,model_path)
    return dest_datas

def read_file(filename):
    with open(filename) as file:
        return file.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into a folder of .npy vector files")
    parser.add_argument('vector_dataset', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('model_path', help='Path to learning model folder.')
    parser.add_argument('output_npy_path', help='Path to output file path where vectors will be stored.')

    args = parser.parse_args()
    proc_path = args.model_path
    final_epoc = read_file(os.path.join(proc_path,"epoc_num.txt"))

    vectors_path = os.path.join(proc_path,"vector_at_{}.npy".format(final_epoc))

    doc_vecs = np.load(vectors_path)

    all_filepaths = [os.path.normpath(fname) for fname in read_file(os.path.join(args.model_path,"music_list.txt")).strip().split("\n")]

    _config = yaml.safe_load(open(args.model_path+"config.yaml"))

    all_word_vecs = calc_all_vectors(all_filepaths,args.vector_dataset,args.model_path,_config)

    summed_word_vecs = np.stack(np.mean(wv,axis=0) for wv in all_word_vecs)
    concat_vecs = np.concatenate([doc_vecs,summed_word_vecs],axis=1)

    np.save(args.output_npy_path,concat_vecs)
