import numpy as np
import random
#import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.signal as tfsignal
import tensorflow as tf
import os
from WeightBias import DenseLayer
from vector_result_processing import ResultsTotal

import process_fma_files
from file_processing import mp3_to_raw_data


CHANCE_SAME_SONG = 0.25

SAMPLERATE = 16000

TRAIN_STEPS_PER_SAVE = 10000000
TRAIN_STEPS_PER_PRINT = 5000

NUM_MUSIC_FILES = 4500
ADAM_learning_rate = 0.001

OUTPUT_VECTOR_SIZE = 32

TIME_SEGMENT_SIZE = 0.1
WINDOW_SIZE = 4

LOWER_EDGE_HERTZ = 80.0
UPPER_EDGE_HERTZ = 7600.0
NUM_MEL_BINS = 64

USE_GPU = False

BATCH_SIZE = 128

STANDARD_SAVE_REPO = "../non-linear-repo/"

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

def find_random_comparison(song_list):
    song_idx = random.randrange(len(song_list))
    song_spec = song_list[song_idx]

    SAME_VAL = 1
    NOT_SAME_VAL = 0
    is_same_val = SAME_VAL if random.random() < CHANCE_SAME_SONG else NOT_SAME_VAL
    numpy_song_idx = np.reshape(np.int32(song_idx),(1,))
    numpy_is_same_val = np.reshape(np.float32(is_same_val),(1,))

    assert len(song_spec) > WINDOW_SIZE*2+1
    choice_origin_idx = random.randrange(WINDOW_SIZE,len(song_spec)-WINDOW_SIZE)

    if random.random() < CHANCE_SAME_SONG:
        offset_idx = choice_origin_idx + random.randint(-WINDOW_SIZE,WINDOW_SIZE)
        compare_vec = song_spec[offset_idx]
    else:
        other_song = random.choice(song_list)
        other_idx = random.randrange(len(other_song))
        compare_vec = song_spec[other_idx]

    return [song_spec[choice_origin_idx],compare_vec,numpy_song_idx,numpy_is_same_val]

def get_train_batch(song_list):
    comparisons = [find_random_comparison(song_list) for _ in range(BATCH_SIZE)]
    origin_batch = np.stack([comp[0] for comp in comparisons])
    compare_batch = np.stack([comp[1] for comp in comparisons])
    song_id_batch = np.stack([comp[2] for comp in comparisons])
    is_same_batch = np.stack([comp[3] for comp in comparisons])
    return origin_batch, compare_batch, song_id_batch, is_same_batch

def load_audio():
    return mp3_to_raw_data("../fma_small/000/000211.mp3",SAMPLERATE)

def plot_spectrogram(Sxx):
    Sxx = Sxx.transpose()
    t = np.arange(Sxx.shape[1])
    f = np.arange(Sxx.shape[0])
    plt.pcolormesh(t, f, Sxx)
    #plt.imshow(Sxx, aspect='auto', cmap='hot_r', origin='lower')
    plt.ylabel('Frequency [bins]')
    plt.xlabel('Time [steps]')
    plt.show()

def calc_spectrogram(raw_sound):
    signals = tf.placeholder(tf.float32, [1, None])
    spectrogram = tf_spectrify(signals)
    with tf.Session() as sess:
        pow_spec_res = sess.run([spectrogram],feed_dict={
            signals: raw_sound.reshape((1,len(raw_sound))),
        })
    return pow_spec_res[0][0]

def save_string(filename,string):
    with open(filename,'w') as file:
        file.write(string)

def save_music_name_list(path_list):
    save_str = "\n".join([os.path.basename(path) for path in path_list])
    save_string(STANDARD_SAVE_REPO+"music_list.txt",save_str)



def tf_spectrify(signals):
    stfts = tf.contrib.signal.stft(signals, frame_length=2**11, frame_step=2**11,
                               fft_length=2**10)
    #power_spectrograms = tf.real(stfts * tf.conj(stfts))
    magnitude_spectrograms = tf.abs(stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      NUM_MEL_BINS, num_spectrogram_bins, SAMPLERATE, LOWER_EDGE_HERTZ,
      UPPER_EDGE_HERTZ)
    mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for <a href="../../api_docs/python/tf/tensordot"><code>tf.tensordot</code></a> does not currently handle this case.
    # mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
    #  linear_to_mel_weight_matrix.shape[-1:]))`
    log_offset = 1e-6
    log_magnitude_spectrograms = tf.log(mel_spectrograms + log_offset)

    return log_magnitude_spectrograms

def spectrify_audios(audio_list):
    signals = tf.placeholder(tf.float32, [1, None])
    spectrogram = tf_spectrify(signals)

    config = tf.ConfigProto(
        device_count = {'GPU': int(False)}
    )
    with tf.Session(config=config) as sess:
        spectrogram_list = []
        for raw_sound in audio_list:
            pow_spec_res = sess.run([spectrogram],feed_dict={
                signals: raw_sound.reshape((1,len(raw_sound))),
            })
            spectrogram_list.append(pow_spec_res[0][0])
        return spectrogram_list

def sqr(x):
    return x * x

def liniearlizer_loss(origin, cross, song_vectors, is_same):
    HIDDEN_LEN = int(NUM_MEL_BINS*1.5)
    origin_next = tf.nn.relu(DenseLayer('relu_origin',NUM_MEL_BINS,HIDDEN_LEN).calc_output(origin))
    origin_vec = tf.nn.sigmoid(DenseLayer('soft_origin',HIDDEN_LEN,OUTPUT_VECTOR_SIZE).calc_output(origin_next))

    cross_next = tf.nn.relu(DenseLayer('relu_cross',NUM_MEL_BINS,HIDDEN_LEN).calc_output(cross))
    cross_vec = tf.nn.sigmoid(DenseLayer('soft_origin',HIDDEN_LEN,OUTPUT_VECTOR_SIZE).calc_output(cross_next))

    return tf.reduce_sum(sqr(cross_vec - origin_vec - song_vectors))

def train_all():
    music_paths, raw_data_list = process_fma_files.get_raw_data_list(SAMPLERATE,num_files=NUM_MUSIC_FILES)
    spectrified_list = spectrify_audios(raw_data_list)

    music_vectors = OutputVectors(len(spectrified_list),OUTPUT_VECTOR_SIZE)

    origin_compare = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_MEL_BINS))
    cross_compare = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_MEL_BINS))
    song_id_batch = tf.placeholder(tf.int32, shape=(BATCH_SIZE, 1))
    is_same_compare = tf.placeholder(tf.float32, shape=(BATCH_SIZE,1))

    global_vectors = music_vectors.get_index_rows(song_id_batch)

    loss = liniearlizer_loss(origin_compare, cross_compare, global_vectors, is_same_compare)

    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    optim = optimizer.minimize(loss)

    result_collection = ResultsTotal(STANDARD_SAVE_REPO)
    weight_saver = tf.train.Saver()

    config = tf.ConfigProto(
        device_count = {'GPU': int(USE_GPU)}
    )

    with tf.Session(config=config) as sess:
        if os.path.exists(STANDARD_SAVE_REPO):
            epoc_start = int(open(STANDARD_SAVE_REPO+"epoc_num.txt").read())
            weight_saver.restore(sess,tf.train.latest_checkpoint(STANDARD_SAVE_REPO))
            save_music_name_list(music_paths)
        else:
            os.makedirs(STANDARD_SAVE_REPO)
            save_music_name_list(music_paths)
            epoc_start = 0
            save_string(STANDARD_SAVE_REPO+"epoc_num.txt",str(epoc_start))
            init = tf.global_variables_initializer()
            sess.run(init)
            weight_saver.save(sess,STANDARD_SAVE_REPO+"savefile",global_step=epoc_start)

        train_steps = 0
        for epoc in range(epoc_start,100000000000):
            epoc_loss_sum = 0
            print("EPOC: {}".format(epoc))
            for x in range(TRAIN_STEPS_PER_SAVE//BATCH_SIZE):
                origin_d, compare_d, song_id_d, is_same_d = get_train_batch(spectrified_list)
                #print()
                opt_res,loss_res = sess.run([optim,loss],feed_dict={
                    origin_compare: origin_d,
                    cross_compare: compare_d,
                    song_id_batch: song_id_d,
                    is_same_compare: is_same_d
                })
                epoc_loss_sum += loss_res
                train_steps += 1
                if train_steps % (TRAIN_STEPS_PER_PRINT//BATCH_SIZE) == 0:
                    print(epoc_loss_sum/(x+1))
            vals = music_vectors.get_vector_values(sess)
            save_string(STANDARD_SAVE_REPO+"epoc_num.txt",str(epoc))
            result_collection.save_file(vals,epoc)
            weight_saver.save(sess,STANDARD_SAVE_REPO+"savefile",global_step=epoc)

#music_paths, raw_data_list = process_fma_files.get_raw_data_list(SAMPLERATE,num_files=NUM_MUSIC_FILES)
#print(get_train_batch(spectrify_audios(raw_data_list)))
#compute_vectors()
#plot_spectrogram(calc_spectrogram(load_audio()))
train_all()
