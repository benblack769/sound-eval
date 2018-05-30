import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Conv1D, Lambda, Concatenate, Reshape
from keras.models import Model
import numpy as np
import tensorflow as tf

from file_processing import mp3_to_raw_data

SAMPLERATE = 16000

BLOCK_SIZE = 1024

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


def shift_1(x):
    val = tf.manip.roll(x,shift=1,axis=1)
    return val

def dialate1_layer(input):
    rolled_layer = Lambda(shift_1)(input)
    concat_input = keras.layers.concatenate([input,rolled_layer])
    mat_input = Reshape((BLOCK_SIZE,2))(concat_input)
    dialte_out = keras.layers.TimeDistributed(Dense(1))(mat_input)
    return dialte_out

def wavenet_model():
    input = Input(shape=(BLOCK_SIZE,))
    output = dialate1_layer(input)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])
    return model

model = wavenet_model()

#import tensorflow as tf
#print("Session\n\n\n\n")
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(sess)
#print("Session\n\n\n\n")

raw_data1 = mp3_to_raw_data('../fma_small/000/000002.mp3')
raw_data2 = mp3_to_raw_data('output.wav')
raw_data = np.concatenate((raw_data1,raw_data2))

batched_data = []
for i in range(0,raw_data.shape[0]-BLOCK_SIZE):
    batched_data.append(raw_data[i:i+BLOCK_SIZE])

#labels = raw_data[BLOCK_SIZE:BLOCK_SIZE+len(batched_data)]
batched_data = np.stack(batched_data)
print(batched_data.shape)

model.fit(batched_data, batched_data, epochs=10, batch_size=20)
