import numpy as np
import random
import tensorflow as tf
import os
import argparse
import shutil
import tempfile
from WeightBias import DenseLayer
from linearlizer import Linearlizer
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor
from helperclasses import OutputVectors,ResultsTotal

import process_many_files
import spectrify
import yaml

config = {} # yaml config info

def save_string(filename,string):
    with open(filename,'w') as file:
        file.write(string)

def save_music_name_list(save_reop,path_list):
    save_str = "\n".join([path for path in path_list])
    save_string(save_reop+"music_list.txt",save_str)

def get_batch_from_var(word_vecs, song_idx, config):
    STEPS_IN_WINDOW = config['STEPS_IN_WINDOW']
    STEPS_BETWEEN_WINDOWS = config['STEPS_BETWEEN_WINDOWS']
    INVALID_BEFORE_RANGE_START = config['INVALID_BEFORE_RANGE_START']
    WINDOW_GAP_RANGE = config['WINDOW_GAP_RANGE']
    INVALID_BEFORE_RANGE_VAR = config['INVALID_BEFORE_RANGE_VAR']
    BATCH_SIZE = config['SAMPLES_PER_SONG']

    word_vecs = word_vecs#[0]
    word_vec_len = tf.shape(word_vecs)[0]

    START_LOC = INVALID_BEFORE_RANGE_START + INVALID_BEFORE_RANGE_VAR + STEPS_IN_WINDOW
    END_LOC = word_vec_len - (STEPS_IN_WINDOW*2 + STEPS_BETWEEN_WINDOWS + WINDOW_GAP_RANGE)
    origin_starts = tf.random_uniform((BATCH_SIZE,1),dtype=tf.int32,minval=START_LOC,maxval=END_LOC)
    valid_compare_starts = origin_starts[:BATCH_SIZE//2] + (STEPS_IN_WINDOW + STEPS_BETWEEN_WINDOWS) + tf.random_uniform((BATCH_SIZE//2,1),dtype=tf.int32,minval=-WINDOW_GAP_RANGE,maxval=WINDOW_GAP_RANGE+1)
    invalid_compare_start = origin_starts[BATCH_SIZE//2:] + (- STEPS_IN_WINDOW - INVALID_BEFORE_RANGE_START) - tf.random_uniform((BATCH_SIZE//2,1),dtype=tf.int32,minval=0,maxval=INVALID_BEFORE_RANGE_VAR)
    all_compare_starts = tf.concat([valid_compare_starts,invalid_compare_start],axis=0)

    arange = tf.range(STEPS_IN_WINDOW)
    tiled_origin_starts = tf.tile(origin_starts,(1,STEPS_IN_WINDOW)) + arange
    tiled_compare_starts = tf.tile(all_compare_starts,(1,STEPS_IN_WINDOW)) + arange

    origins = tf.gather(word_vecs, tiled_origin_starts,axis=0)
    compares = tf.gather(word_vecs, tiled_compare_starts,axis=0)

    valid_is_trues = tf.zeros((BATCH_SIZE//2,),dtype=tf.float32)
    invalid_is_trues = tf.ones((BATCH_SIZE//2,),dtype=tf.float32)
    is_correct = tf.concat([valid_is_trues,invalid_is_trues],axis=0)

    #tiled_song_idx = tf.tile(tf.reshape(song_idx,(1,)),(BATCH_SIZE,))

    return origins,compares,song_idx,is_correct

def make_file_generator(all_filenames, source_dir):
    def file_generator():
        file_indicies = np.arange(len(all_filenames))
        for _ in range(100000):
            np.random.shuffle(file_indicies)
            for idx in file_indicies:
                np_data = np.load(source_dir+all_filenames[idx])
                if len(np_data) > 50:
                    yield (np_data, idx)
    return file_generator

def flatten_batch(batch, desired_shapes):
    res = []
    for item,shape in zip(batch,desired_shapes):
        res.append(tf.reshape(item,shape))
    return res

def get_read_npy(base_folder):
    def read_npy(fname):
        return np.load(base_folder+fname.decode())
    return read_npy

def train_all():
    SAVE_REPO = config['STANDARD_SAVE_REPO']

    all_filenames = music_paths = process_many_files.get_all_paths(config['BASE_MUSIC_FOLDER'],"npy",lambda fname: np.load(fname).shape[0] > 100)
    #print(all_filenames)
    actual_generator = make_file_generator(all_filenames,config['BASE_MUSIC_FOLDER'])
    #fnames = tf.data.Dataset.from_tensor_slices(all_filenames)
    #fnames = fnames.shuffle(len(all_filenames))
    #act_read_npy = get_read_npy(config['BASE_MUSIC_FOLDER'])
    #ds1 = fnames.map(
    #    lambda item: tuple(tf.py_func(act_read_npy, [item], [tf.float32,])),num_parallel_calls=multiprocessing.cpu_count())
    #ds2 = tf.data.Dataset.from_tensor_slices(tf.range(len(all_filenames)))
    #ds3 = tf.data.Dataset.from_tensor_slices(get_np_lens(all_filenames))
    #ds = tf.data.Dataset.zip((ds1,ds2))

    #all_data = list(actual_generator())#()]
    ds = tf.data.Dataset.from_generator(
        actual_generator, (tf.float32,tf.int32), (tf.TensorShape([None,config['NUM_MEL_BINS']]),tf.TensorShape([])))
    #ds1 = tf.data.Dataset.from_tensor_slices([tf.constant(data[0]) for data in all_data])
    #ds2 = tf.data.Dataset.from_tensors([tf.constant(data[1]) for data in all_data])
    #ds3 = tf.data.Dataset.from_tensors([tf.constant(data[2]) for data in all_data])
    #ds = tf.data.Dataset.zip([ds1,ds2,ds3])
    ds = ds.map(lambda word_vecs,song_idx: get_batch_from_var(word_vecs,song_idx,config),num_parallel_calls=multiprocessing.cpu_count())
    ds = ds.repeat(count=10000000000000)

    SAMPLE_SIZE = config['SAMPLES_PER_SONG']
    STEPS_IN_WINDOW = config['STEPS_IN_WINDOW']
    VEC_SIZE = config['NUM_MEL_BINS']
    SONGS_PER_BATCH =  config['SONGS_PER_BATCH']
    FINAL_SIZE = SAMPLE_SIZE * SONGS_PER_BATCH
    desired_shapes = (
        [FINAL_SIZE,STEPS_IN_WINDOW,VEC_SIZE],
        [FINAL_SIZE,STEPS_IN_WINDOW,VEC_SIZE],
        [SONGS_PER_BATCH],
        [FINAL_SIZE],
        )
    #,multiprocessing.cpu_count())
    ds = ds.batch(SONGS_PER_BATCH)
    #ds = ds.map(lambda a1,a2,a3,a4: flatten_batch([a1,a2,a3,a4],desired_shapes))
    ds = ds.prefetch(8)

    iter = ds.make_one_shot_iterator()
    origin_compare, cross_compare, song_idxs, is_same_compare = iter.get_next()

    music_vectors = OutputVectors(len(all_filenames),config['OUTPUT_VECTOR_SIZE'])

    song_vecs = music_vectors.get_index_rows(song_idxs)
    tiled_song_vecs = tf.reshape(song_vecs,(SONGS_PER_BATCH,1,config['OUTPUT_VECTOR_SIZE']))
    tiled_song_vecs = tf.tile(tiled_song_vecs,(1,SAMPLE_SIZE,1))

    input_shape = (SONGS_PER_BATCH,SAMPLE_SIZE,STEPS_IN_WINDOW,VEC_SIZE)
    linearlizer = Linearlizer(config['NUM_MEL_BINS'], config['STEPS_IN_WINDOW'], config['HIDDEN_SIZE'], config['OUTPUT_VECTOR_SIZE'], input_shape, config['MAG_REGULARIZATION_PARAM'])

    loss = linearlizer.loss(origin_compare, cross_compare, tiled_song_vecs, is_same_compare)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=SGD_learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate=config['ADAM_learning_rate'])
    optimizer = tf.train.RMSPropOptimizer(learning_rate=config['ADAM_learning_rate'],epsilon=10e-5)
    optim = optimizer.minimize(loss)

    result_collection = ResultsTotal(SAVE_REPO)

    gpuconfig = tf.ConfigProto(
        device_count = {'GPU': int(config['USE_GPU'])}
    )
    with tf.Session(config=gpuconfig) as sess:
        if os.path.exists(SAVE_REPO):
            epoc_start = int(open(SAVE_REPO+"epoc_num.txt").read())
            save_music_name_list(SAVE_REPO,music_paths)
            init = tf.global_variables_initializer()
            sess.run(init)
            linearlizer.load(sess,SAVE_REPO)
            music_vectors.load_vector_values(sess,result_collection.load_file(epoc_start))
        else:
            os.makedirs(SAVE_REPO)
            open(SAVE_REPO+"cost_list.csv",'w').write("epoc,cost\n")
            save_music_name_list(SAVE_REPO,music_paths)
            epoc_start = 0
            save_string(SAVE_REPO+"epoc_num.txt",str(epoc_start))
            init = tf.global_variables_initializer()
            sess.run(init)
            linearlizer.save(sess,SAVE_REPO)
            result_collection.save_file(music_vectors.get_vector_values(sess),epoc_start)

        shutil.copy(config['CONFIG_PATH'],SAVE_REPO+"config.yaml")

        train_steps = 0
        for epoc in range(epoc_start,100000000000):
            epoc_loss_sum = 0
            print("EPOC: {}".format(epoc))
            for x in range(config['TRAIN_STEPS_PER_SAVE']//FINAL_SIZE):
                #print()
                opt_res,loss_res = sess.run([optim,loss])
                epoc_loss_sum += loss_res
                train_steps += 1
                if train_steps % (config['TRAIN_STEPS_PER_PRINT']//FINAL_SIZE) == 0:
                    print(epoc_loss_sum/(x+1))
            save_string(SAVE_REPO+"epoc_num.txt",str(epoc))
            result_collection.save_file(music_vectors.get_vector_values(sess),epoc)
            linearlizer.save(sess,SAVE_REPO)
            open(SAVE_REPO+"cost_list.csv",'a').write("{},{}\n".format(epoc,epoc_loss_sum/(x+1)))


#music_paths, raw_data_list = process_many_files.get_raw_data_list(SAMPLERATE,num_files=NUM_MUSIC_FILES)
#print(get_train_batch(spectrify_audios(raw_data_list)))
#compute_vectors()
#plot_spectrogram(calc_spectrogram(load_audio()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .npy identify files into document vectors")
    parser.add_argument('vector_dataset', help='Path to folder full of .npy files (looks recursively into subfolders for .npy files).')
    parser.add_argument('output_folder', help='Path to output folder where files will be stored.')
    parser.add_argument('--config', dest='config_yaml', default="default_config.yaml",
                    help='define the .yaml config file (default is "defaultconfig.yaml")')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_yaml))

    config['STANDARD_SAVE_REPO'] = args.output_folder
    config['BASE_MUSIC_FOLDER'] = args.vector_dataset
    config['CONFIG_PATH'] = args.config_yaml

    train_all()
