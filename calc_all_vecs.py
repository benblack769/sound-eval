import numpy as np
import argparse
import yaml
import os
import process_many_files
import tensorflow as tf
from linearlizer import Linearlizer

def run_spec_list(spec_list,config,model_path):
    vecs = tf.placeholder(tf.float32, [None, config['NUM_MEL_BINS']])

    #vecs = tf.reshape(vecs,(None, config['NUM_MEL_BINS'], 1))

    SAMPLE_SIZE = config['SAMPLES_PER_SONG']
    STEPS_IN_WINDOW = config['STEPS_IN_WINDOW']
    VEC_SIZE = config['NUM_MEL_BINS']
    SONGS_PER_BATCH =  config['SONGS_PER_BATCH']

    input_shape = (-1,STEPS_IN_WINDOW,VEC_SIZE)

    batches = [vecs[x:tf.shape(vecs)[0]+x-STEPS_IN_WINDOW] for x in range(STEPS_IN_WINDOW)]

    inputs = tf.stack(batches,axis=2)
    print(inputs.shape)

    linearlizer = Linearlizer(config['NUM_MEL_BINS'], STEPS_IN_WINDOW, config['HIDDEN_SIZE'], config['OUTPUT_VECTOR_SIZE'], input_shape, config['MAG_REGULARIZATION_PARAM'])

    wordvec = linearlizer.word_vector(inputs)
    with tf.Session() as sess:
        linearlizer.load(sess, model_path)
        for spec in spec_list:
            val = sess.run(wordvec,feed_dict={
                vecs:spec
            })
            yield val

def make_dirs(paths):
    for path in paths:
        pathdir = os.path.dirname(path)
        if not os.path.exists(pathdir):
            os.makedirs(pathdir)

def calc_all_vectors(source_dir, dest_dir, model_path, config):
    all_filenames = process_many_files.get_all_paths(source_dir,"npy",lambda fname: np.load(fname).shape[0] > 100)
    source_abs_filenames = [os.path.join(source_dir,filename) for filename in all_filenames]
    dest_abs_filenames = [os.path.join(dest_dir,filename) for filename in all_filenames]

    make_dirs(dest_abs_filenames)
    source_datas = (np.load(path) for path in source_abs_filenames)
    dest_datas = run_spec_list(source_datas,config,model_path)
    for dpath,ddata in zip(dest_abs_filenames,dest_datas):
        np.save(dpath,ddata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into a folder of .npy vector files")
    parser.add_argument('vector_dataset', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('model_path', help='Path to learning model folder.')
    parser.add_argument('output_folder', help='Path to output folder where files will be stored.')

    args = parser.parse_args()

    _config = yaml.safe_load(open(args.model_path+"config.yaml"))

    calc_all_vectors(args.vector_dataset,args.output_folder,args.model_path,_config)
