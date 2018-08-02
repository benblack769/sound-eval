import tensorflow as tf
from WeightBias import DenseLayer
import numpy as np
import os

class Linearlizer:
    def __init__(self,
            input_size,
            hidden_size,
            output_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_origin = DenseLayer('relu_origin',input_size,hidden_size)
        self.hidden_cross = DenseLayer('relu_cross',input_size,hidden_size)
        self.out_origin = DenseLayer('soft_origin',hidden_size,output_size)
        self.out_cross = DenseLayer('soft_cross',hidden_size,output_size)

    def loss(self, origin, cross, song_vectors, is_same):
        origin_vec = self.word_vector(origin)

        cross_next = tf.nn.relu(self.hidden_cross.calc_output(cross))
        cross_vec = self.out_cross.calc_output(cross_next)

        input_vec = origin_vec + song_vectors
        output_vec = cross_vec
        logit_assignment = tf.nn.sigmoid(tf.reduce_mean(input_vec * output_vec,axis=1)*0.1)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_assignment,labels=is_same)
        return tf.reduce_mean(cost)

    def word_vector(self, input):
        origin_next = tf.nn.relu(self.hidden_origin.calc_output(input))
        origin_vec = self.out_origin.calc_output(origin_next)
        return origin_vec

    def vars(self):
        return (
            self.hidden_origin.wb_list() +
            self.hidden_cross.wb_list() +
            self.out_origin.wb_list() +
            self.out_cross.wb_list()
        )

    def load(self, sess, folder):
        for var in self.vars():
            name = os.path.join(folder, var.name+".npy")
            value = np.load(name)
            var.load(value, sess)

    def save(self, sess, folder):
        for var in self.vars():
            value = sess.run(var)
            name =  os.path.join(folder, var.name+".npy")
            print(name)
            np.save(name, value)
