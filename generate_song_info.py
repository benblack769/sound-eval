import numpy as np
import random
import os
import argparse
import shutil
import pandas

def labels_to_numbers(labels):
    all_label_ids = set(labels)
    mapping = {lab:idx for idx,lab in enumerate(all_label_ids)}
    numbers = [mapping[lab] for lab in labels]
    return np.asarray(numbers,dtype=np.int32)

def one_hot(nums):
    max_size = np.amax(nums)+1
    dense_vec = np.zeros([len(nums),max_size],dtype=np.float32)
    dense_vec[np.arange(len(nums)),nums] = 1
    return dense_vec

def set_ones(numbers,max_size):
    vec = np.zeros(max_size,dtype=np.bool_)
    vec[numbers] = 1
    return vec

def generate_label_vectors(label_df,sample_size,key_name):
    assert sample_size <= len(label_df)
    row_numbers = np.random.choice(len(label_df),size=sample_size,replace=False)
    is_selelected = np.equal(np.arange(len(label_df)), set_ones(row_numbers,len(label_df)))
    selected_rows = label_df.loc[is_selelected]
    label_nums = labels_to_numbers(label_df[key_name])
    one_hot_labels = one_hot(label_nums)
    id_ordering = list([str(id) for id in label_df['item_id']])
    return one_hot_labels,is_selelected,id_ordering

def main_process(genre_csv_fname,output_folder_fname,sample_size,key_name):
    tracks_csv = pandas.read_csv(genre_csv_fname)

    label_vec,is_labeled_vec,id_ordering = generate_label_vectors(tracks_csv,sample_size,key_name)

    if os.path.exists(output_folder_fname):
        shutil.rmtree(output_folder_fname)
    os.mkdir(output_folder_fname)

    LABELS_PATH = os.path.join(output_folder_fname,"labels.npy")
    IS_LABELED_PATH = os.path.join(output_folder_fname,"is_labeled.npy")
    MUSIC_IDS_PATH = os.path.join(output_folder_fname,"music_ids.txt")

    np.save(LABELS_PATH,label_vec)
    np.save(IS_LABELED_PATH,is_labeled_vec)

    with open(MUSIC_IDS_PATH,'w') as mus_list_file:
        mus_list_file.write("\n".join(id_ordering))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into document vectors")
    parser.add_argument('genre_csv', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('output_folder', help='Path to output folder where files will be stored.')
    parser.add_argument('--sample_size', dest='sample_size', default="50",
                    help='number of songs to label')
    parser.add_argument('--key_name', dest='key_name', default="genre",
                    help='key of entry in dataframe')
    args = parser.parse_args()

    main_process(args.genre_csv,args.output_folder,int(args.sample_size),args.key_name)
