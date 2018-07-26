import numpy as np
import random
import tensorflow as tf
import os
from scipy import signal

import process_fma_files
from file_processing import mp3_to_raw_data


SAMPLERATE = 16000

NUM_MUSIC_FILES = 5000
ADAM_learning_rate = 0.001

TIME_SEGMENT_SIZE = 0.1

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

def load_audio():
    return mp3_to_raw_data("../fma_small/000/000211.mp3",SAMPLERATE)

def spectrify(raw_sound):
    f, t, Sxx  = signal.spectrogram(raw_sound,fs=SAMPLERATE,nperseg=2**6)
    by_time = Sxx.transpose()

    seconds_in_song = len(raw_sound) / SAMPLERATE
    frames_per_sec = Sxx.shape[1] / seconds_in_song
    return 

print(spectrify(load_audio()))
