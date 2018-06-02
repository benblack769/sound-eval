import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Conv1D, Lambda, Concatenate, Reshape
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import tensorflow as tf
#import theano

from file_processing import mp3_to_raw_data, raw_data_to_wav
from process_fma_files import get_raw_data

SAMPLERATE = 16000

BLOCK_SIZE = 20000

def seq_model():
    model = Sequential()
    model.add(Dense(20, input_dim=BLOCK_SIZE))
    model.add(Activation('relu'))
    model.add(Dense(BLOCK_SIZE))
    model.add(Activation('tanh'))

    model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])
    return model


def shift_x(shift_ammount):
    def shift_x_fn(x):
        val = tf.manip.roll(x,shift=shift_ammount,axis=1)
        return val
    return shift_x_fn

def dist_dense(flat_vec):
    in_mat = Reshape((BLOCK_SIZE,1))(flat_vec)
    time_mat = keras.layers.TimeDistributed(Dense(1))(in_mat)
    return  Reshape((BLOCK_SIZE,))(time_mat)

def dialate_layer(input, dialate_length):
    rolled_layer = Lambda(shift_x(BLOCK_SIZE-dialate_length))(input)
    concat_input = keras.layers.Concatenate()([Reshape((BLOCK_SIZE,1))(input),Reshape((BLOCK_SIZE,1))(rolled_layer)])
    mat_input = Reshape((BLOCK_SIZE,2))(concat_input)
    dialte_out = keras.layers.TimeDistributed(Dense(1,use_bias=True))(mat_input)
    arg = Reshape((BLOCK_SIZE,))(dialte_out)
    return arg#Activation('relu')(arg)

#def dilated_layer(x):

def wavenet_model():
    input = Input(shape=(BLOCK_SIZE,))
    di1 = dialate_layer(input,1)
    di1 = dialate_layer(di1,1)
    di2 = dialate_layer(di1,2)
    di4 = dialate_layer(di2,4)
    di8 = dialate_layer(di4,8)
    l1_m = keras.layers.Multiply()([Activation('sigmoid')(di8),Activation('tanh')(di8)])
    l1_f = dist_dense(l1_m)
    l1_r = keras.layers.Add()([l1_f,input])
    di1 = dialate_layer(l1_r,1)
    di2 = dialate_layer(di1,2)
    di4 = dialate_layer(di2,4)
    di8 = dialate_layer(di4,8)
    l2_m = keras.layers.Multiply()([Activation('sigmoid')(di8),Activation('tanh')(di8)])
    l2_f = dist_dense(l2_m)
    #l1_r = keras.layers.Add()([l2_f,input])
    #di1 = dialate_layer(l2_r,1)
    #di2 = dialate_layer(di1,2)
    #di4 = dialate_layer(di2,4)
    model = Model(inputs=input, outputs=l2_f)
    model.compile(optimizer='Adam',
              loss='mean_absolute_error',
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
raw_data = get_raw_data(SAMPLERATE)
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

for _ in range(10):
    model.fit(batched_data, pred_data, batch_size=32)
model.save_weights('arg.h5')

model.load_weights('arg.h5')
pred_data = model.predict(batched_data,batch_size=32)
out_raw = deblockify_data(pred_data)
raw_data_to_wav('sing_channel.wav',raw_data,SAMPLERATE)
raw_data_to_wav('processed.wav',out_raw,SAMPLERATE)

#model.save_weights('my_model_weights.h5')

#model.save("weights.h5")
