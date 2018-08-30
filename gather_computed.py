import numpy as np
import argparse
import os

def read_file(filename):
    with open(filename) as file:
        return file.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into a folder of .npy vector files")
    parser.add_argument('vector_dataset', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('vector_list', help='filename of list of filenames that correspond to output order.')
    parser.add_argument('output_npy_path', help='Path to output file path where vectors will be stored.')

    args = parser.parse_args()

    source_dir = args.vector_dataset
    all_filepaths = [os.path.normpath(fname) for fname in read_file(args.vector_list).strip().split("\n")]
    source_abs_filenames = [os.path.join(source_dir,filename) for filename in all_filepaths]

    summed_word_vecs = np.stack(np.mean(np.load(fname),axis=0) for fname in source_abs_filenames)
    #concat_vecs = np.concatenate([doc_vecs,summed_word_vecs],axis=1)

    np.save(args.output_npy_path,summed_word_vecs)
