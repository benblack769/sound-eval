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
from process_fma_files import get_raw_data

SAMPLERATE = 10000

BLOCK_SIZE = 2000

LAYER_WIDTH = 4

OUTPUT_SIZE = 16

NUM_MUSIC_FOLDERS = 1

SONG_SAMPLES = 5 # number of samples from same song

def shift_x(shift_ammount):
    def shift_x_fn(x):
        # the 0 axis here is the batch axis, so the 1 axis
        # is the 0 axis to kerasa
        tensorflow_0_axis = 2
        val = tf.manip.roll(x,shift=shift_ammount,axis=tensorflow_0_axis)
        return val
    return shift_x_fn

def select_second_last_elmt(tensor):
    return tensor[:,:,-1]

def select_second_last_elmt_shape(input_shape):
    return tuple(input_shape[0:2] + [input_shape[3]])

def dialate_layer(input, dialate_length, input_size=LAYER_WIDTH, output_size=LAYER_WIDTH):
    rolled_layer = Lambda(shift_x(-dialate_length))(input)
    concat_input = keras.layers.Concatenate(axis=2)([input,rolled_layer])
    mat_input = concat_input
    sigmoid_linear_part = keras.layers.TimeDistributed(Dense(LAYER_WIDTH),input_shape=(input_size))(mat_input)
    tanh_linear_part = keras.layers.TimeDistributed(Dense(LAYER_WIDTH),input_shape=(input_size))(mat_input)
    print("\n\n",sigmoid_linear_part.shape,"\n\n")
    print("\n\n",mat_input.shape,"\n\n")
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

def wavenet_model():
    NUM_INPUTS = SONG_SAMPLES + 1
    input = Input(shape=(NUM_INPUTS,BLOCK_SIZE))
    shape_input = Reshape((NUM_INPUTS,BLOCK_SIZE,1))(input)
    final_res1,out1 = dialate_layer(shape_input,1,1,LAYER_WIDTH)
    final_res_l1,out2 = incr_pow2(out1,7)
    final_res_l2,out3 = incr_pow2(out2,8)
    #final_res_l3,out4 = incr_pow2(out3,9)
    #final_res_l4,out5 = incr_pow2(out4,9)
    final_result_concat = keras.layers.Concatenate(axis=3)(
        [final_res1]
        + final_res_l1
        + final_res_l2
        #+ final_res_l3
        #+ final_res_l4
    )
    first_elmt_in_concat = Lambda(select_second_last_elmt,output_shape=select_second_last_elmt_shape(final_result_concat.shape))(final_result_concat)
    print(first_elmt_in_concat.shape)

    final_value = Dense(1)(first_elmt_in_concat)
    final_value_shape = Reshape((BLOCK_SIZE,))(final_value)
    #l1_r = keras.layers.Add()([l2_f,input])
    #di1 = dialate_layer(l2_r,1)
    #di2 = dialate_layer(di1,2)
    #di4 = dialate_layer(di2,4)
    model = Model(inputs=input, outputs=final_value_shape)
    model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
    return model

model = wavenet_model()

#plot_model(model, to_file='model.png')
#exit(1)
#import tensorflow as tf
#print("Session\n\n\n\n")
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(sess)
#print("Session\n\n\n\n")
    #raw_data2 = mp3_to_raw_data('output.wav',SAMPLERATE)
    #raw_data = np.concatenate((raw_data1,raw_data2))
raw_data = get_raw_data(SAMPLERATE,NUM_MUSIC_FOLDERS)
print(raw_data.shape)

def blockify_data(r_data):
    batched_data = []
    for i in range(0,r_data.shape[0]-BLOCK_SIZE,BLOCK_SIZE//2):
        batched_data.append(raw_data[i:i+BLOCK_SIZE])

    #labels = raw_data[BLOCK_SIZE:BLOCK_SIZE+len(batched_data)]
    batched_data = np.stack(batched_data)
    return batched_data

def deblockify_data(b_data):
    raw_blocks = b_data[:,:BLOCK_SIZE//2]
    return raw_blocks.flatten()


batched_data = blockify_data(raw_data)
pred_data = np.roll(batched_data,shift=1,axis=1)
#print(pred_data[30][0:20])
#print(batched_data[30][00:20])
#print(batched_data.shape)
#exit(1)

#for _ in range(10):

model.fit(batched_data, pred_data,
    batch_size=32,
    epochs=20)

model.save_weights('arg.h5')

model.load_weights('arg.h5')
pred_data = model.predict(batched_data,batch_size=32)
out_raw = deblockify_data(pred_data)

SECTION_SIZE = 1000

class Trainer:
    def __init__(self, model):
        self.model = model

    def train_item(self, same_song_section_list, compare_song_section, is_frome_same_song):
        # trains item
        input = np.stack(same_song_section_list+[compare_song_section])
        print(input.shape)
        label = np.float32(is_frome_same_song)

        model.fit(input, label,
            batch_size=1,
            epochs=1,
            )

    def batch_train_item(self,same_song_section_list,different_song_section):
        # puts items on queue for later batched training
        # if queue reaches certain size, then execute the batch and train
        pass

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

    def get_sections_in_song(self, song_id, num_sections):
        song_size = len(self.song_list[song_id])
        #max_sections_in_song = len(self.song_list[song_id]) / SECTION_SIZE
        #assert num_sections < max_sections_in_song
        MAX_NUM_TRIES = 200
        for i in range(MAX_NUM_TRIES):
            section_split = np.random.randint(low=0,high=song_size-SECTION_SIZE-1,size=num_sections)
            did_overlap = self.section_split_not_overlapping(section_split)
            if not did_overlap:
                return [self.song_section(song_id,start) for start in section_split]
            elif i == MAX_NUM_TRIES-1:
                assert False, "tried to split song into too many sections. \nPerhaps a very short song in is the list?"

    def song_section(self, song_id, start_loc):
        song = self.song_list[song_id]
        assert len(song) >= start_loc+SECTION_SIZE
        return song[start_loc:start_loc+SECTION_SIZE]

    def section_split_not_overlapping(self, section_split):
        sorted_splits = np.sort(section_split)
        sorted_split_ends = sorted_splits + SECTION_SIZE
        did_overlap = sorted_split_ends[1:] < sorted_splits[:-1]
        return did_overlap.any()




def prune_output_10m(total_output):
    ten_min_samples = 60*10*SAMPLERATE
    return total_output[:ten_min_samples] if len(total_output) > ten_min_samples else total_output

raw_data_to_wav('sing_channel.wav',prune_output_10m(raw_data),SAMPLERATE)
raw_data_to_wav('processed.wav',prune_output_10m(out_raw),SAMPLERATE)

#model.save_weights('my_model_weights.h5')

#model.save("weights.h5")
