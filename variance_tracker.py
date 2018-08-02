import numpy as np
import random
import tensorflow as tf
import os
from WeightBias import DenseLayer
from linearlizer import Linearlizer
import matplotlib.pyplot as plt
from spectrify import calc_spectrogram, plot_spectrogram

from file_processing import mp3_to_raw_data
from audio_config import *

SGD_learn_rate = 1.0

def plot_line(time_points,line_data,y_axis_label):
    plt.plot(time_points,line_data)
    plt.ylabel(y_axis_label)
    plt.xlabel('Time [seconds]')
    plt.show()

def make_batch_for(all_word_vecs,all_cross_vecs,timestep):
    orign_vec = all_word_vecs[timestep]
    cross_vec_1 = all_cross_vecs[timestep-1]
    cross_vec_2 = np.sum(all_cross_vecs,axis=0)/(len(all_cross_vecs)*1000)

    is_same = np.asarray([1,0])
    origin = np.stack([orign_vec,orign_vec])
    cross = np.stack([cross_vec_1,cross_vec_2])
    return origin, cross, is_same

def calc_all_vectors(calc_fn, spectrogram, sess):
    audio_batch = tf.placeholder(tf.float32, shape=(len(spectrogram), NUM_MEL_BINS))
    out_vecs = calc_fn(audio_batch)
    res = sess.run(out_vecs, feed_dict={
        audio_batch:spectrogram
    })
    print
    return res

def get_glob_diffs(glob_states):
    np_states = np.stack(glob_states)
    diff_vals = np_states[1:] - np_states[:-1]
    diff_amnts = np.sum(np.abs(diff_vals),axis=1)
    return diff_amnts

def plot_track_music_fns(mp3_path):
    raw_sound = mp3_to_raw_data(mp3_path,SAMPLERATE)
    spectrogram = calc_spectrogram(raw_sound,NUM_MEL_BINS,SAMPLERATE,TIME_SEGMENT_SIZE)
    linearlizer = Linearlizer(NUM_MEL_BINS,HIDDEN_SIZE,OUTPUT_VECTOR_SIZE)

    moving_glob_init = np.random.standard_normal((OUTPUT_VECTOR_SIZE)).astype('float32')/OUTPUT_VECTOR_SIZE
    moving_glob_vector = tf.Variable(moving_glob_init,name="output_vectors")

    word_vec = tf.placeholder(tf.float32, shape=(2, OUTPUT_VECTOR_SIZE))
    cmp_vec = tf.placeholder(tf.float32, shape=(2, OUTPUT_VECTOR_SIZE))
    is_same = tf.placeholder(tf.float32, shape=(2,))
    stacked_glob = tf.stack([moving_glob_vector,moving_glob_vector])

    loss = linearlizer.loss_vec_computed(word_vec, cmp_vec, stacked_glob, is_same)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=SGD_learn_rate)
    optim = optimizer.minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        linearlizer.load(sess, STANDARD_SAVE_REPO)
        all_word_vecs = calc_all_vectors(linearlizer.word_vector,spectrogram,sess)
        all_cross_vecs = calc_all_vectors(linearlizer.compare_vector,spectrogram,sess)
        losses = []
        glob_states = []
        glob_states.append(sess.run(moving_glob_vector))
        for step in range(1,len(all_word_vecs)-1):
            word,cmp,same = make_batch_for(all_word_vecs,all_cross_vecs,step)
            opt_val, loss_val = sess.run([optim,loss],feed_dict={
                word_vec:word,
                cmp_vec:cmp,
                is_same:same,
            })
            print(loss_val)
            losses.append(loss_val)

            glob_states.append(sess.run(moving_glob_vector))
            #print(word)
            #print(sess.run(moving_glob_vector))

        glob_diffs = get_glob_diffs(glob_states)

        time_line = np.arange(0,len(glob_diffs))*TIME_SEGMENT_SIZE
        plot_spectrogram(spectrogram,TIME_SEGMENT_SIZE)
        plot_line(time_line, glob_diffs, "glob diffs (abs)")
        plot_line(time_line, losses, "losses (cross entropy)")


plot_track_music_fns("../fma_small/000/000197.mp3")
