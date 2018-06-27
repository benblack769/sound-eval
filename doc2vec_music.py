import numpy as np
import random
import tensorflow as tf
import os

import process_fma_files
from tf_standard_repo_wavenet import *
from vector_result_processing import ResultsTotal

#from WeightBias import DenseLayer
#from fast_wavenet_model import Model, OutputVectors
#from wavenet import WaveNetModel, mu_law_decode, mu_law_encode

#from pixnn import discretized_mix_logistic_loss
SAMPLERATE = 16000

TRAIN_STEPS_PER_SAVE = 2000

NUM_MUSIC_FILES = 2000
ADAM_learning_rate = 0.001

np.set_printoptions(floatmode='fixed')

class OutputVectors:
    def __init__(self,num_songs,vector_size):
        init_val_flat = np.random.randn(num_songs*vector_size).astype('float32')
        init_val = np.reshape(init_val_flat,(num_songs,vector_size))

        self.num_songs = num_songs
        self.vector_size = vector_size
        self.all_vectors = tf.Variable(init_val,name="output_vectors")

    def get_index_rows(self,indicies):
        return tf.reshape(tf.gather(self.all_vectors,indicies),shape=(indicies.shape[0],self.vector_size))

    def get_vector_values(self,sess):
        return sess.run(self.all_vectors)

class SectionGenerator:
    def __init__(self, song_list):
        self.song_list = song_list

    def random_song_id(self):
        return random.randrange(0,len(self.song_list))

    def random_section_in_song(self, song_id):
        song_size = len(self.song_list[song_id])
        return self.song_section(song_id,random.randint(0,song_size-BLOCK_SIZE-1))

    def song_section(self, song_id, start_loc):
        song = self.song_list[song_id]
        assert len(song) >= start_loc+BLOCK_SIZE
        return song[start_loc:start_loc+BLOCK_SIZE]

def save_string(filename,string):
    with open(filename,'w') as file:
        file.write(string)

def get_train_batch(song_list):
    sec_gen = SectionGenerator(song_list)
    batch_songs = [sec_gen.random_song_id() for _ in range(BATCH_SIZE)]
    batch_list = [sec_gen.random_section_in_song(song) for song in batch_songs]
    batch_matrix = np.stack(batch_list)
    batch_song_indicies = np.asarray(batch_songs,dtype=np.int32)
    return batch_song_indicies,batch_matrix

def save_music_name_list(path_list):
    save_str = "\n".join([os.path.basename(path) for path in path_list])
    save_string(SONG_VECTOR_SIZE+"music_list.txt",save_str)

def train_all():
    music_paths, raw_data_list = process_fma_files.get_raw_data_list(SAMPLERATE,num_files=NUM_MUSIC_FILES)

    music_vectors = OutputVectors(len(raw_data_list),SONG_VECTOR_SIZE)


    audio_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, BLOCK_SIZE))
    gc_id_batch = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))

    global_vectors = music_vectors.get_index_rows(gc_id_batch)

    loss = wavenet_loss(audio_batch,global_vectors)

    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    optim = optimizer.minimize(loss)

    result_collection = ResultsTotal(STANDARD_SAVE_REPO)
    weight_saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': int(USE_GPU)}
    )

    with tf.Session(config=config) as sess:
        save_music_name_list(music_paths)
        if os.path.exists(STANDARD_SAVE_REPO):
            epoc_start = int(open(STANDARD_SAVE_REPO+"epoc_num.txt").read())
            weight_saver.restore(sess,tf.train.latest_checkpoint(STANDARD_SAVE_REPO))
        else:
            os.makedirs(STANDARD_SAVE_REPO)
            epoc_start = 0
            init = tf.global_variables_initializer()
            sess.run(init)

        for epoc in range(epoc_start,100000000000):
            epoc_loss_sum = 0
            print("EPOC: {}".format(epoc))
            for x in range(TRAIN_STEPS_PER_SAVE//BATCH_SIZE):
                batch_song_indicies, batch_input = get_train_batch(raw_data_list)
                opt_res,loss_res = sess.run([optim,loss],feed_dict={
                    audio_batch: batch_input,
                    gc_id_batch: batch_song_indicies
                })
                epoc_loss_sum += loss_res
                print(epoc_loss_sum/(x+1))
            vals = music_vectors.get_vector_values(sess)
            save_string(STANDARD_SAVE_REPO+"epoc_num.txt",str(epoc))
            result_collection.save_file(vals,epoc)
            weight_saver.save(sess,STANDARD_SAVE_REPO+"savefile",global_step=epoc)

train_all()
