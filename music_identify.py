import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Conv1D, Lambda, Concatenate, Reshape
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import math
import random
import tensorflow as tf

from file_processing import mp3_to_raw_data, raw_data_to_wav
import process_fma_files

SAMPLERATE = 10000

BLOCK_SIZE = 1000
SECTION_SIZE = BLOCK_SIZE

LAYER_WIDTH = 16
#HIDDEN_SIZE = 48

NUM_SONGS = 50

BATCH_SIZE = 4

NUM_BATCHES_PER_KERAS = 500

SONG_SAMPLES = 1 # number of samples from same song

TRAINING_SAMPLES = 2000000

def shift_x(shift_ammount):
    def shift_x_fn(x):
        # the 0 axis here is the batch axis, so the 1 axis
        # is the 0 axis to kerasa
        tensorflow_0_axis = 2
        val = tf.manip.roll(x,shift=shift_ammount,axis=tensorflow_0_axis)
        return val
    return shift_x_fn


def int_shape(x):
    return list(map(int, x.get_shape()))

def d_to_int(x):
    return int(str(x)) if x and str(x) != "?" else None

def select_second_last_elmt(tensor):
    tshape =  tensor[:,:,-1]
    print(tshape.shape)
    return tshape

def select_second_last_elmt_shape(input_shape):
    return (d_to_int(input_shape[0]), d_to_int(input_shape[1]), d_to_int(input_shape[3]))

def dialate_layer(input, dialate_length, input_size=LAYER_WIDTH, output_size=LAYER_WIDTH):
    rolled_layer = Lambda(shift_x(-dialate_length))(input)
    concat_input = keras.layers.Concatenate(axis=3)([input,rolled_layer])
    mat_input = concat_input
    sigmoid_linear_part = keras.layers.TimeDistributed(Dense(output_size),input_shape=(input_size,))(mat_input)
    tanh_linear_part = keras.layers.TimeDistributed(Dense(output_size),input_shape=(input_size,))(mat_input)
    l1_m = keras.layers.Multiply()([Activation('sigmoid')(sigmoid_linear_part),Activation('tanh')(tanh_linear_part)])
    #l1_f = dist_dense(l1_m)
    l1_r = keras.layers.Add()([l1_m,input]) if input_size == output_size else l1_m
    return l1_m,l1_r#Activation('relu')(arg)

#def dilated_layer(x):

def incr_pow2(input, highest_pow2):
    final_list = []
    cur_input = input
    for i in range(0,highest_pow2+1):
        skip_len = 2**i
        final_res2,out2 = dialate_layer(cur_input,1)
        cur_input = out2
        final_list.append(final_res2)
    return final_list, cur_input

def mse_only_on_label(y_true,y_pred):
    return keras.backend.square(y_pred[0] - y_true[0])

def wavenet_model():
    NUM_INPUTS = SONG_SAMPLES + 1
    input = Input(shape=(NUM_INPUTS,BLOCK_SIZE))
    shape_input = Reshape((NUM_INPUTS,BLOCK_SIZE,1))(input)
    final_res1,out1 = dialate_layer(shape_input,1,1,LAYER_WIDTH)
    final_res_l1,out2 = incr_pow2(out1,7)
    final_res_l2,out3 = incr_pow2(out2,8)
    final_res_l3,out4 = incr_pow2(out3,8)
    #final_res_l3,out4 = incr_pow2(out3,9)
    #final_res_l4,out5 = incr_pow2(out4,9)
    final_result_concat = keras.layers.Add()( #keras.layers.Concatenate(axis=3)(
        [final_res1]
        + final_res_l1
        + final_res_l2
        + final_res_l3
        #+ final_res_l4
    )
    last_elmt_in_concat = Lambda(select_second_last_elmt,output_shape=select_second_last_elmt_shape(final_result_concat.shape))(final_result_concat)
    #print(last_elmt_in_concat.shape)

    def sum_all_but_last(lst_emt):
        #print("lstsmmd")
        #print(lst_emt.shape)
        sum_same_song = tf.reduce_sum(lst_emt[:,:SONG_SAMPLES],axis=1)
        comparitor = lst_emt[:,SONG_SAMPLES]
        #print(comparitor.shape)
        #print(sum_same_song.shape)
        concatted = tf.concat([sum_same_song,comparitor],axis=1)
        #print(concatted.shape)
        return concatted

    #summed_last = Lambda(sum_all_but_last,output_shape=(2*d_to_int(last_elmt_in_concat.shape[2]),))(last_elmt_in_concat)
    flattened_last = Reshape((d_to_int(last_elmt_in_concat.shape[1])*d_to_int(last_elmt_in_concat.shape[2]),))(last_elmt_in_concat)

    # linear evaluation might actually work better
    hidden_value = flattened_last#Dense(HIDDEN_SIZE, activation='relu')(summed_last)

    final_value = Dense(1,activation='sigmoid')(hidden_value)

    #l1_r = keras.layers.Add()([l2_f,input])
    #di1 = dialate_layer(l2_r,1)
    #di2 = dialate_layer(di1,2)
    #di4 = dialate_layer(di2,4)
    #flattened_last_element = Reshape(((SONG_SAMPLES+1)*LAYER_WIDTH,))(last_elmt_in_concat)
    #concat_output = keras.layers.Concatenate()([final_value,flattened_last_element])
    model = Model(inputs=input, outputs=final_value)
    model.compile(optimizer='Adam',
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
    return model

#model = wavenet_model()
#exit(1)

#plot_model(model, to_file='model.png')
#exit(1)
#import tensorflow as tf
#print("Session\n\n\n\n")
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(sess)
#print("Session\n\n\n\n")
    #raw_data2 = mp3_to_raw_data('output.wav',SAMPLERATE)
    #raw_data = np.concatenate((raw_data1,raw_data2))





class Trainer:
    def __init__(self, model):
        self.model = model
        self.batch_labels = []
        self.batch_input = []

    def train_item(self, same_song_section_list, compare_song_section, is_frome_same_song):
        # trains item
        batch_ready_input, batch_ready_label = self.format_train_items(same_song_section_list, compare_song_section, is_frome_same_song)
        self.model.fit(batch_ready_input, batch_ready_label,
            batch_size=1,
            epochs=1,
            )

    def format_train_items(self,same_song_section_list, compare_song_section, is_frome_same_song):
        input = np.stack(same_song_section_list+[compare_song_section])
        batch_ready_input = np.reshape(input,(1,)+input.shape)
        label = np.float32(is_frome_same_song)
        reshaped_label = np.reshape(label,(1,1))
        #padded_label = np.zeros((1,(SONG_SAMPLES+1)*LAYER_WIDTH))
        #batch_ready_label = np.concatenate([reshaped_label,padded_label],axis=1)
        return batch_ready_input,reshaped_label

    def batch_train_item(self,same_song_section_list,compare_song_section,is_frome_same_song):
        # puts items on queue for later batched training
        # if queue reaches certain size, then execute the batch and train
        batch_ready_input, batch_ready_label = self.format_train_items(same_song_section_list, compare_song_section, is_frome_same_song)
        self.batch_labels.append(batch_ready_label)
        self.batch_input.append(batch_ready_input)

        assert len(self.batch_labels) == len(self.batch_input)
        if len(self.batch_labels) >= BATCH_SIZE*NUM_BATCHES_PER_KERAS:
            batched_input = np.concatenate(self.batch_input)
            batched_labels = np.concatenate(self.batch_labels)
            self.model.fit(batched_input, batched_labels,
                batch_size=BATCH_SIZE,
                epochs=1,
                shuffle=False,
                )
            self.batch_labels = []
            self.batch_input = []


    def eval_section(self, section):
        # evaluates and returns rep vector of section
        pass


class SectionGenerator:
    '''
    Gets a list of single-channel song numpy vectors
    get_sections_in_song evalues Returns sections of those samples
    '''
    def __init__(self, song_list):
        self.song_list = song_list

    def random_song_id(self):
        return random.randrange(0,len(self.song_list))

    def random_song_other_than(self, other_song_id):
        s_id = random.randrange(0,len(self.song_list))
        while s_id == other_song_id:
            s_id = random.randrange(0,len(self.song_list))
        return s_id

    def get_sequential_sections_in_song(self, song_id, num_sections):
        song_size = len(self.song_list[song_id])
        #max_sections_in_song = len(self.song_list[song_id]) / SECTION_SIZE
        #assert num_sections < max_sections_in_song
        get_block_size = num_sections * SECTION_SIZE
        assert song_size > get_block_size
        section_start = random.randint(0,song_size-get_block_size)
        return [self.song_section(song_id, start*SECTION_SIZE + section_start) for start in range(0, num_sections)]

    def get_section_in_song(self, song_id):
        return self.get_sequential_sections_in_song(song_id,1)[0]

    def song_section(self, song_id, start_loc):
        song = self.song_list[song_id]
        assert len(song) >= start_loc+SECTION_SIZE
        return song[start_loc:start_loc+SECTION_SIZE]

def prune_output_10m(total_output):
    ten_min_samples = 60*10*SAMPLERATE
    return total_output[:ten_min_samples] if len(total_output) > ten_min_samples else total_output

def train_full(model,music_list):
    sec_gen = SectionGenerator(music_list)
    trainer = Trainer(model)
    is_same_value = 1.0
    is_not_same_value = 0.0
    for _ in range(TRAINING_SAMPLES):
        song_id = sec_gen.random_song_id()
        # trains with samples from same song
        song_samples = sec_gen.get_sequential_sections_in_song(song_id,SONG_SAMPLES+1)
        trainer.batch_train_item(song_samples[:SONG_SAMPLES],song_samples[SONG_SAMPLES],is_same_value)

        # trains with samples from different songs
        other_song_id = sec_gen.random_song_other_than(song_id)
        assert other_song_id != song_id
        #new_song_samples = sec_gen.get_sequential_sections_in_song(song_id,SONG_SAMPLES)
        other_song_sample = sec_gen.get_section_in_song(other_song_id)
        trainer.batch_train_item(song_samples[:SONG_SAMPLES],other_song_sample,is_not_same_value)

def output_all_weights(model, music_list, path_list):
    pass

def full_run():
    model = wavenet_model()
    music_paths,music_list = process_fma_files.get_raw_data_list(SAMPLERATE,NUM_SONGS)
    train_full(model,music_list)
    model.save_weights('../identify_weights.h5')

full_run()

#raw_data_to_wav('sing_channel.wav',prune_output_10m(raw_data),SAMPLERATE)
#raw_data_to_wav('processed.wav',prune_output_10m(out_raw),SAMPLERATE)

#model.save_weights('my_model_weights.h5')

#model.save("weights.h5")
