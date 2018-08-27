import tensorflow as tf
from WeightBias import DenseLayer
import numpy as np
import os

def sqr(x):
    return x * x

class Linearlizer:
    def __init__(self,
            input_size,
            input_width,
            hidden_size,
            output_size,
            input_shape):

        self.input_size = input_size
        self.input_dims = input_dims = input_size * input_width
        self.input_width = input_width
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_shape = input_shape
        HIDDEN_SIZE_2 = 256

        self.hidden_origin = DenseLayer('relu_origin',input_dims,hidden_size)
        #self.hidden2_origin = DenseLayer('relu2_origin',hidden_size,HIDDEN_SIZE_2)
        self.out_origin = DenseLayer('soft_origin',hidden_size,output_size)

        self.hidden_cross = DenseLayer('relu_cross',input_dims,hidden_size)
        #self.hidden2_cross = DenseLayer('relu2_cross',hidden_size,HIDDEN_SIZE_2)
        self.out_cross = DenseLayer('soft_cross',hidden_size,output_size)

    def loss(self, origin, cross, song_vectors, is_same):
        return self.loss_vec_computed(self.word_vector(origin), self.compare_vector(cross), song_vectors, is_same)

    def loss_vec_computed(self, word_vector, cross_vector, global_vector, is_same):
        input_vec = word_vector + global_vector
        output_vec = cross_vector

        reg_cost = 0.005 * (tf.reduce_mean(sqr(input_vec)) + tf.reduce_mean(sqr(output_vec)))

        logit_assignment = tf.nn.sigmoid(tf.reduce_mean(input_vec * output_vec,axis=2)*0.1)
        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_assignment,labels=is_same)
        return tf.reduce_mean(cost) + reg_cost

    def word_vector(self, input):
        input = tf.reshape(input,shape=self.input_shape[:-2]+(self.input_dims,))
        origin_next = tf.nn.relu(self.hidden_origin.calc_output(input))
        #origin_next = tf.nn.relu(self.hidden2_origin.calc_output(origin_next))
        origin_vec = self.out_origin.calc_output(origin_next)
        return origin_vec

    def compare_vector(self, input_cmp):
        input_cmp = tf.reshape(input_cmp,shape=self.input_shape[:-2]+(self.input_dims,))
        #input_cmp = tf.reshape(input_cmp,(input_cmp.shape[0],input_cmp.shape[1]*input_cmp.shape[2]))
        cross_next = tf.nn.relu(self.hidden_cross.calc_output(input_cmp))
        #cross_next = tf.nn.relu(self.hidden2_cross.calc_output(cross_next))
        cross_vec = self.out_cross.calc_output(cross_next)
        return cross_vec

    def vars(self):
        return (
            self.hidden_origin.wb_list() +
            self.hidden_cross.wb_list() +
            self.out_origin.wb_list() +
            self.out_cross.wb_list()
        )

    def load(self, sess, folder):
        for var in self.vars():
            save_var_name = var.name[:-2]
            name = os.path.join(folder, save_var_name+".npy")
            value = np.load(name)
            var.load(value, sess)

    def save(self, sess, folder):
        for var in self.vars():
            save_var_name = var.name[:-2]
            value = sess.run(var)
            name =  os.path.join(folder, save_var_name+".npy")
            np.save(name, value)
