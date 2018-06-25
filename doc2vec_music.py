import numpy as np
import random
import tensorflow as tf

import process_fma_files
from tf_fast_wavenet import *

#from WeightBias import DenseLayer
#from fast_wavenet_model import Model, OutputVectors
#from wavenet import WaveNetModel, mu_law_decode, mu_law_encode

#from pixnn import discretized_mix_logistic_loss
SAMPLERATE = 16000

TRAIN_STEPS = 10

NUM_MUSIC_FILES = 8
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

def get_train_batch(song_list):
    sec_gen = SectionGenerator(song_list)
    batch_songs = [sec_gen.random_song_id() for _ in range(BATCH_SIZE)]
    batch_list = [sec_gen.random_section_in_song(song) for song in batch_songs]
    batch_matrix = np.stack(batch_list)
    batch_song_indicies = np.asarray(batch_songs,dtype=np.int32)
    return batch_song_indicies,batch_matrix

#def expected_from_input(batched_input):
#    return np.roll(batched_input,shift=1,axis=1)

def train_all():
    music_paths, raw_data_list = process_fma_files.get_raw_data_list(SAMPLERATE,num_files=NUM_MUSIC_FILES)

    music_vectors = OutputVectors(len(raw_data_list),SONG_VECTOR_SIZE)

    audio_batch = tf.placeholder(tf.float32, shape=(BATCH_SIZE, BLOCK_SIZE))
    gc_id_batch = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))

    global_vectors = music_vectors.get_index_rows(gc_id_batch)

    loss = wavenet_loss(audio_batch,global_vectors)

    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    optim = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoc in range(100):
            for x in range(TRAIN_STEPS//BATCH_SIZE):
                batch_song_indicies, batch_input = get_train_batch(raw_data_list)
                sess.run(optim,feed_dict={
                    audio_batch: batch_input,
                    gc_id_batch: batch_song_indicies
                })
            vals = music_vectors.get_vector_values(sess)
            np.save("arg.npy",vals)
            print(vals)

train_all()
exit(0)
def get_dist_train_pred_fns(inputs,targets):
    HIDDEN_SIZE_1 = 30
    HIDDEN_SIZE_2 = 70
    ADAM_learning_rate = 0.001


    in_to_hid_mul = DenseLayer("in_to_hid_mul",1,HIDDEN_SIZE_1)
    in_to_hid_tanh = DenseLayer("in_to_hid_tanh",1,HIDDEN_SIZE_1)
    hid1_to_hid2_mul = DenseLayer("hid1_to_hid2_mul",HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    hid1_to_hid2_tanh = DenseLayer("hid1_to_hid2_tanh",HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    hid_to_out = DenseLayer("hid_to_out",HIDDEN_SIZE_2,NUM_BUCKETS)

    hid_layer_1_mul = tf.sigmoid(in_to_hid_mul.calc_output(inputs))
    hid_layer_1_tanh = tf.nn.relu(in_to_hid_tanh.calc_output(inputs))
    hid_layer_1_res = tf.multiply(hid_layer_1_mul,hid_layer_1_tanh)

    hid_layer_2_mul = tf.sigmoid(hid1_to_hid2_mul.calc_output(hid_layer_1_res))
    hid_layer_2_tanh = tf.nn.relu(hid1_to_hid2_tanh.calc_output(hid_layer_1_res))
    hid_layer_2_res = tf.multiply(hid_layer_2_mul,hid_layer_2_tanh)

    out_layer = tf.sigmoid(hid_to_out.calc_output(hid_layer_2_res))

    final_cost = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = targets,
        logits = out_layer,
    )

    prediction = tf.nn.softmax(out_layer)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=ADAM_learning_rate,momentum=0.9)

    train_op = optimizer.minimize(final_cost)
    return prediction,train_op

def randomly_pull_ammount(nparr,batch_size):
    indicies = np.random.choice(len(nparr), batch_size)
    return np.stack(nparr[idx] for idx in indicies)

def run_training(sess,train_op,inputs,targets):
    BATCH_SIZE = 32

    train_input = np.reshape(norm_data,(len(norm_data),1))
    train_expected = numbers_to_vectors(train_input)

    for epoc in range(10):
        for x in range(0,len(train_input)-BATCH_SIZE,BATCH_SIZE):
            sess.run(train_op,feed_dict={
                inputs: randomly_pull_ammount(train_input,BATCH_SIZE),
                targets: randomly_pull_ammount(train_expected,BATCH_SIZE),
            })

def run_test(sess,pred_op,inputs,targets):
    test_input = np.reshape(test_data,(len(test_data),1))
    test_expected = numbers_to_vectors(test_input)
    sample = 5
    result = sess.run(pred_op,feed_dict={
        inputs: test_input[sample:sample+5],
        targets: test_expected[sample:sample+5],
    })
    print(result)




def run_all():
    inputs = tf.placeholder(tf.float32, shape=(None, 1))
    targets = tf.placeholder(tf.float32, shape=(None, NUM_BUCKETS))

    pred_op,train_op = get_dist_train_pred_fns(inputs,targets)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("initted",flush=True)
        run_training(sess,train_op,inputs,targets)
        print("training finished",flush=True)
        run_test(sess,pred_op,inputs,targets)

run_all()

exit(0)
