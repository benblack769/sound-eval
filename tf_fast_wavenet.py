import tensorflow as tf
from WeightBias import DenseLayer

HIDDEN_SIZE_1 = 50
HIDDEN_SIZE_2 = 52

def wavenet_loss(input_batch,global_vector_batch):
    input_block = tf.concat([input_batch[:,:-1],global_vector_batch],axis=1)

    hidden_fn_1 = DenseLayer("a",int(input_block.shape[1]),HIDDEN_SIZE_1)
    hidden_fn_2 = DenseLayer("b",HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    output_fn = DenseLayer("c",HIDDEN_SIZE_2,1)

    hidden1 = tf.tanh(hidden_fn_1.calc_output(input_block))
    hidden2 = tf.tanh(hidden_fn_2.calc_output(hidden1))

    output = tf.sigmoid(output_fn.calc_output(hidden2))

    loss = tf.square(output - input_batch[:,-1])
    return loss


def tensorflow_wavenet_loss(input_batch,global_vector_batch):
    input_block = tf.concat([input_batch[:,:-1],global_vector_batch],axis=1)

    hidden_fn_1 = DenseLayer("a",int(input_block.shape[1]),HIDDEN_SIZE_1)
    hidden_fn_2 = DenseLayer("b",HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    output_fn = DenseLayer("c",HIDDEN_SIZE_2,1)

    hidden1 = tf.tanh(hidden_fn_1.calc_output(input_block))
    hidden2 = tf.tanh(hidden_fn_2.calc_output(hidden1))

    output = tf.sigmoid(output_fn.calc_output(hidden2))

    loss = tf.square(output - input_batch[:,-1])
    return loss
