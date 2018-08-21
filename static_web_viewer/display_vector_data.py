import sklearn.manifold
import numpy as np
import pandas
import argparse
import subprocess
import os
import shutil
import math
import random
import tempfile
import json
import multiprocessing
import recursive_html_indexing

MP3_FOLDER = "mp3_files/"
OUT_DATAFRAME = "all_data.csv"
OUT_JSON_VIEW = "all_data.json"
MP3_VECTOR_LIST = "mp3_vecs.npy"
VEC_JSON = "vec_json.json"
OUT_DATAFRAME_PART = "view_data.csv"
OUT_JSON_PART_VIEW = "view_data.json"
VIEWER_JSON = "template_json.js"
VIEWER_HTML = "display_template.html"
VIEWER_JS = "template.js"

def all_mp3_files(root_dir):
    find_call = ["find", root_dir, "-name", '*.mp3', "-type", "f" ]
    filename_list = subprocess.check_output(find_call).decode('utf-8').split("\n")[:-1]
    return filename_list

def make_dir_overwrite(name):
    if os.path.exists(name):
        if os.path.isdir(name):
            shutil.rmtree(name)
        else:
            os.remove(name)
    os.mkdir(name)

def init_dirs(output_path):
    make_dir_overwrite(output_path)


def build_cosine_dist_matrix(data):
    fdata = np.float32(data)
    fone = np.float32(1.0001)

    sqr_data = np.sqrt(np.sum(fdata*fdata,axis=1))

    top_data = np.stack(
        np.sum(fdata[i] * fdata,axis=1) for i in range(len(data))
    )
    bottom_data = np.reshape(sqr_data,(len(data),1)) * sqr_data.transpose()

    full_comp = fone-(top_data / bottom_data)

    return full_comp

def calc_tsne(data):
    import sys
    print("tsne started")
    cos_dist_matrix = build_cosine_dist_matrix(data)
    print(cos_dist_matrix.shape)
    sys.stdout.flush()
    #def cosine_d(d1,d2):
    #    return 1.0 - np.sum(d1*d2) / (math.sqrt(np.sum(d1*d1)) * math.sqrt(np.sum(d2*d2)))
    tsne = sklearn.manifold.TSNE(metric="precomputed")
    #print(data.shape)
    #exit(1)
    #distance_matrix = pairwise_distances(data, data, metric='cosine', n_jobs=1)
    transformed_data = tsne.fit_transform(cos_dist_matrix)
    print(transformed_data.shape)
    print("tsne ended")
    sys.stdout.flush()
    return transformed_data
    #print(data)


def associate_metadata(data_2d, associate_dataframe, actual_filenames):
    xvals,yvals = np.transpose(data_2d)
    #print(xvals)
    #print(actual_filenames)
    val_dataframe = pandas.DataFrame(data={
        "filename":actual_filenames,
        "x":xvals,
        "y":yvals
    })
    unique_ass_data = associate_dataframe.drop_duplicates(subset="filename")
    joined_metadata = val_dataframe.merge(unique_ass_data,on="filename",how="left",sort=False)

    return joined_metadata

def read_file(filename):
    with open(filename) as file:
        return file.read()


def prepare_json_var(json_name,js_name):
    with open(js_name,'w') as js_file:
        js_file.write("var input_json_data = " + read_file(json_name))

def save_string(filename, string):
    with open(filename, 'w') as file:
        file.write(string)

def save_doc_data(output_path,associated_data,filenames_list,doc_vecs):
    np.save(os.path.join(output_path,MP3_VECTOR_LIST),doc_vecs)

    tranformed_data = calc_tsne(doc_vecs)

    out_dataframe = associate_metadata(tranformed_data,associated_data,filenames_list)

    save_string(os.path.join(output_path,VEC_JSON),json.dumps(round_list_lists(doc_vecs.tolist()),separators=(',', ':')))

    out_dataframe.to_csv(os.path.join(output_path,OUT_DATAFRAME),index=False)
    out_dataframe.to_json(os.path.join(output_path,OUT_JSON_VIEW),orient="records")

    copy_view_files(output_path)
    #prepare_json_var(os.path.join(output_path,OUT_JSON_VIEW),os.path.join(output_path,VIEWER_JSON))

def copy_safe(src,dest):
    if os.path.exists(src):
        dir = os.path.dirname(dest)
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src,dest)

def copy_mp3s(mp3_base_filepath, output_paths, copy_paths):
    for item in copy_paths:
        copy_safe(os.path.join(mp3_base_filepath,item),os.path.join(output_paths,item))

def copy_view_files(output_path):
    source_folder = os.path.dirname(__file__)+"/"
    shutil.copyfile(source_folder+"viewer/display_template.html",os.path.join(output_path,VIEWER_HTML))
    shutil.copyfile(source_folder+"viewer/template.js",os.path.join(output_path,VIEWER_JS))
    shutil.copyfile(source_folder+"viewer/math_lib.js",os.path.join(output_path,"math_lib.js"))
    shutil.copyfile(source_folder+"viewer/metricsgraphics.js",os.path.join(output_path,"metricsgraphics.js"))

def round_list_lists(lls):
    return [[round(x,8) for x in l] for l in lls]

def are_unique(items):
    return len(items) == len(set(items))

def order_dataframe_by_filelist(df,file_list):
    assert are_unique(file_list), "file lists must be unique"
    fdf = pandas.DataFrame({"filename":file_list})
    df.filename = df.filename
    merged = fdf.merge(df,on=["filename"],how="left")
    #print(len(merged.filename))
    return merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .mid files into csvs with relevant data")
    parser.add_argument('doc2vec_result_folder', help='Path to spectrogram doc2vec output folder')
    parser.add_argument('mp3_dataset_root', help='Path to folder full of .mp3 files (needs to be the same structure as the folder that spectrogram doc2vec processed)')
    parser.add_argument('dataset_csv', help='Path to csv file with associated data of dataset')
    parser.add_argument('output_folder', help='Path to static website output')
    parser.add_argument('--vectors_npy', dest='vectors_fname', default="default",
                help='override npy vectors to use as base vectors')

    args = parser.parse_args()

    output_path = args.output_folder
    #copy_view_files(output_path)
    mp3_dataset_root = args.mp3_dataset_root #"../../../../Downloads/midiworld_mp3/"
    proc_path = args.doc2vec_result_folder

    final_epoc = read_file(os.path.join(proc_path,"epoc_num.txt"))

    vectors_path = args.vectors_fname if args.vectors_fname != "default" else os.path.join(proc_path,"vector_at_{}.npy".format(final_epoc))
    csv_path = args.dataset_csv

    actual_vecs = np.load(vectors_path)
    add_data = pandas.read_csv(csv_path)
    all_mp3_filepaths = [os.path.normpath(fname)[:-4] for fname in read_file(os.path.join(proc_path,"music_list.txt")).strip().split("\n")]

    #ordered_add_data = order_dataframe_by_filelist(add_data,all_mp3_filepaths)
    #print(add_data)
    #print(ordered_add_data)

    init_dirs(output_path)

    save_doc_data(output_path,add_data,all_mp3_filepaths,actual_vecs)

    copy_mp3s(mp3_dataset_root,os.path.join(output_path,MP3_FOLDER),all_mp3_filepaths)

    recursive_html_indexing.indexify_folder(output_path)
