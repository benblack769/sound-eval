import numpy as np
import random
import tensorflow as tf
import os
import argparse
import shutil
from WeightBias import DenseLayer
from linearlizer import Linearlizer

import process_many_files
from file_processing import mp3_to_raw_data
from spectrify import spectrify_audios
import yaml

config = {} # yaml config info


class OutputVectors:
    def __init__(self,num_songs,vector_size):
        init_val = np.random.standard_normal((num_songs,vector_size)).astype('float32')/vector_size

        self.num_songs = num_songs
        self.vector_size = vector_size
        self.all_vectors = tf.Variable(init_val,name="output_vectors")

    def get_index_rows(self,indicies):
        return tf.reshape(tf.gather(self.all_vectors,indicies),shape=(indicies.shape[0],self.vector_size))

    def get_vector_values(self,sess):
        return sess.run(self.all_vectors)

    def load_vector_values(self,sess,values):
        sess.run(self.all_vectors.assign(values))

class ResultsTotal:
    def __init__(self,vectors_dir):
        self.vectors_dir = vectors_dir

    def get_filepath(self,timestep):
        return "{path}vector_at_{timestep}.npy".format(path=self.vectors_dir,timestep=timestep)

    def load_file(self,timestep):
        return np.load(self.get_filepath(timestep))

    def save_file(self,data,timestep):
        np.save(self.get_filepath(timestep),data)

    def clear_files(self):
        shutil.rmtree(self.vectors_dir)

def save_string(filename,string):
    with open(filename,'w') as file:
        file.write(string)

def save_music_name_list(save_reop,path_list):
    save_str = "\n".join([path for path in path_list])
    save_string(save_reop+"music_list.txt",save_str)

def get_batch_from_var(flat_spectrified_var, song_start_markers, all_song_lens, BATCH_SIZE, WINDOW_SIZE):
    num_time_slots = flat_spectrified_var.shape[0]
    song_ids = tf.random_uniform((BATCH_SIZE,),dtype=tf.int32,minval=0,maxval=song_start_markers.shape[0])
    song_start_vals = tf.gather(song_start_markers,song_ids,axis=0)
    song_lens = tf.gather(all_song_lens,song_ids,axis=0)
    add_vals_float = tf.cast(song_lens,dtype=tf.float32)*tf.random_uniform((BATCH_SIZE,),dtype=tf.float32,minval=0,maxval=1)
    add_vals_int = tf.cast(add_vals_float,dtype=tf.int32)

    base_time_slot_ids = song_start_vals + add_vals_int

    compare_valid_ids = base_time_slot_ids[:BATCH_SIZE//2] + tf.random_uniform((BATCH_SIZE//2,),dtype=tf.int32,minval=-WINDOW_SIZE,maxval=WINDOW_SIZE+1)
    compare_valid_ids = tf.maximum(np.int32(0),tf.minimum(num_time_slots-1,compare_valid_ids))
    compare_invalid_ids = tf.random_uniform((BATCH_SIZE//2,),dtype=tf.int32,minval=0,maxval=num_time_slots)
    compare_ids = tf.concat([compare_valid_ids,compare_invalid_ids],axis=0)

    valid_is_trues = tf.zeros((BATCH_SIZE//2,),dtype=tf.float32)
    invalid_is_trues = tf.ones((BATCH_SIZE//2,),dtype=tf.float32)
    is_correct = tf.concat([valid_is_trues,invalid_is_trues],axis=0)

    orign_vecs = tf.gather(flat_spectrified_var,base_time_slot_ids,axis=0)
    compare_vecs = tf.gather(flat_spectrified_var,compare_ids,axis=0)
    print(orign_vecs.shape)
    return orign_vecs,compare_vecs,song_ids,is_correct

def save_reference_vecs(save_repo,spectrified_list,max_reference_vecs):
    # Used by variance_tracker.py
    if len(spectrified_list) > max_reference_vecs:
        selected_list = spectrified_list[np.random.choice(len(spectrified_list),size=max_reference_vecs,replace=False)]
    else:
        selected_list = spectrified_list

    np.save(save_repo+"reference_vecs.npy",selected_list)

def flatten_audios(spec_list):
    lens = [len(spec) for spec in spec_list]
    begin_lens = [0]+lens[:-1]
    np_lens = np.asarray(begin_lens)
    start_markers = np.cumsum(np_lens)
    flattened_specs = np.concatenate(spec_list)
    return flattened_specs, start_markers, np.asarray(lens)

def train_all():
    SAVE_REPO = config['STANDARD_SAVE_REPO']
    BATCH_SIZE = config['BATCH_SIZE']

    music_paths, raw_data_list = process_many_files.get_raw_data_list(config['SAMPLERATE'], config['BASE_MUSIC_FOLDER'], max_num_files=config['MAX_NUM_FILES'])
    spectrified_list = spectrify_audios(raw_data_list,config['NUM_MEL_BINS'], config['SAMPLERATE'], config['TIME_SEGMENT_SIZE'])
    #spectrified_list = crop_to_smallest(spectrified_list)

    #num_song_ids = spectrified_list.shape[0] * spectrified_list.shape[1]
    #flat_spectrified_list = spectrified_list.reshape((num_song_ids,spectrified_list.shape[2]))
    flat_spectrified_list, flat_start_markers, song_lens = flatten_audios(spectrified_list)

    tf_song_lens = tf.constant(song_lens,dtype=tf.int32)
    tf_flat_start_markers = tf.constant(flat_start_markers,dtype=tf.int32)
    flat_spectrified_var = tf.constant(flat_spectrified_list,dtype=tf.float32)

    music_vectors = OutputVectors(len(spectrified_list),config['OUTPUT_VECTOR_SIZE'])

    origin_compare, cross_compare, song_id_batch, is_same_compare = get_batch_from_var(flat_spectrified_var, tf_flat_start_markers, tf_song_lens, BATCH_SIZE, config['WINDOW_SIZE'])

    global_vectors = music_vectors.get_index_rows(song_id_batch)

    linearlizer = Linearlizer(config['NUM_MEL_BINS'], config['HIDDEN_SIZE'], config['OUTPUT_VECTOR_SIZE'])

    loss = linearlizer.loss(origin_compare, cross_compare, global_vectors, is_same_compare)

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=SGD_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=config['ADAM_learning_rate'])
    optim = optimizer.minimize(loss)

    result_collection = ResultsTotal(SAVE_REPO)

    with tf.Session() as sess:
        if os.path.exists(SAVE_REPO):
            epoc_start = int(open(SAVE_REPO+"epoc_num.txt").read())
            save_music_name_list(SAVE_REPO,music_paths)
            init = tf.global_variables_initializer()
            sess.run(init)
            linearlizer.load(sess,SAVE_REPO)
            music_vectors.load_vector_values(sess,result_collection.load_file(epoc_start))
        else:
            os.makedirs(SAVE_REPO)
            save_music_name_list(SAVE_REPO,music_paths)
            epoc_start = 0
            save_string(SAVE_REPO+"epoc_num.txt",str(epoc_start))
            init = tf.global_variables_initializer()
            sess.run(init)
            linearlizer.save(sess,SAVE_REPO)
            result_collection.save_file(music_vectors.get_vector_values(sess),epoc_start)

        save_reference_vecs(SAVE_REPO,flat_spectrified_list,config['NUM_REFERENCE_VECS'])
        shutil.copy(config['CONFIG_PATH'],SAVE_REPO+"config.yaml")

        train_steps = 0
        for epoc in range(epoc_start,100000000000):
            epoc_loss_sum = 0
            print("EPOC: {}".format(epoc))
            for x in range(config['TRAIN_STEPS_PER_SAVE']//BATCH_SIZE):
                #print()
                opt_res,loss_res = sess.run([optim,loss])
                epoc_loss_sum += loss_res
                train_steps += 1
                if train_steps % (config['TRAIN_STEPS_PER_PRINT']//BATCH_SIZE) == 0:
                    print(epoc_loss_sum/(x+1))
            save_string(SAVE_REPO+"epoc_num.txt",str(epoc))
            result_collection.save_file(music_vectors.get_vector_values(sess),epoc)
            linearlizer.save(sess,SAVE_REPO)

#music_paths, raw_data_list = process_many_files.get_raw_data_list(SAMPLERATE,num_files=NUM_MUSIC_FILES)
#print(get_train_batch(spectrify_audios(raw_data_list)))
#compute_vectors()
#plot_spectrogram(calc_spectrogram(load_audio()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn a folder full of .mp3 files into vectors")
    parser.add_argument('mp3_dataset', help='Path to folder full of .mp3 files (looks recursively into subfolders for .mp3 files).')
    parser.add_argument('output_folder', help='Path to output folder where files will be stored.')
    parser.add_argument('--config', dest='config_yaml', default="default_config.yaml",
                    help='define the .yaml config file (default is "default_config.yaml")')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_yaml))
    print(config)
    config['STANDARD_SAVE_REPO'] = args.output_folder
    config['BASE_MUSIC_FOLDER'] = args.mp3_dataset
    config['CONFIG_PATH'] = args.config_yaml

    train_all()
