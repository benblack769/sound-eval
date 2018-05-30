import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Conv1D, Lambda, Concatenate, Reshape
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import tensorflow as tf
#import theano

from file_processing import mp3_to_raw_data

SAMPLERATE = 16000

BLOCK_SIZE = 512

def seq_model():
    model = Sequential()
    model.add(Dense(20, input_dim=BLOCK_SIZE))
    model.add(Activation('relu'))
    model.add(Dense(1))
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

def dialate_layer(input, dialate_length):
    rolled_layer = Lambda(shift_x(BLOCK_SIZE-dialate_length))(input)
    concat_input = keras.layers.Concatenate()([Reshape((BLOCK_SIZE,1))(input),Reshape((BLOCK_SIZE,1))(rolled_layer)])
    mat_input = Reshape((BLOCK_SIZE,2))(concat_input)
    dialte_out = keras.layers.TimeDistributed(Dense(1,use_bias=True))(mat_input)
    arg = Reshape((BLOCK_SIZE,))(dialte_out)
    return arg

def wavenet_model():
    input = Input(shape=(BLOCK_SIZE,))
    di1 = dialate_layer(input,1)
    di2 = dialate_layer(di1,2)
    model = Model(inputs=input, outputs=di2)
    model.compile(optimizer='rmsprop',
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

raw_data1 = mp3_to_raw_data('../fma_small/000/000002.mp3')
raw_data2 = mp3_to_raw_data('output.wav')
raw_data = np.concatenate((raw_data1,raw_data2))

batched_data = []
for i in range(0,raw_data.shape[0]-BLOCK_SIZE,BLOCK_SIZE//2):
    batched_data.append(raw_data[i:i+BLOCK_SIZE])

#labels = raw_data[BLOCK_SIZE:BLOCK_SIZE+len(batched_data)]
batched_data = np.stack(batched_data)
pred_data = np.roll(batched_data,shift=2,axis=1)
print(pred_data[30][0:20])
print(batched_data[30][00:20])
print(batched_data.shape)
#exit(1)

for _ in range(10):
    model.fit(batched_data, pred_data, batch_size=10)
#model.save("weights.h5")
