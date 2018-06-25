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


def shift_x(x,shift_ammount):
    # the 0 axis here is the batch axis, so the 1 axis
    # is the 0 axis to kerasa
    tensorflow_0_axis = 1
    val = tf.manip.roll(x,shift=shift_ammount,axis=tensorflow_0_axis)
    return val

def dialate_layer(input, dialate_length, input_size=LAYER_WIDTH, output_size=LAYER_WIDTH):
    rolled_layer = shift_x(input,-dialate_length)
    concat_input = keras.layers.Concatenate(axis=2)([input,rolled_layer])
    mat_input = concat_input
    sigmoid_linear_part = keras.layers.TimeDistributed(Dense(LAYER_WIDTH))(mat_input)
    tanh_linear_part = keras.layers.TimeDistributed(Dense(LAYER_WIDTH))(mat_input)
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
    input = Input(shape=(BLOCK_SIZE,))
    shape_input = Reshape((BLOCK_SIZE,1))(input)
    final_res1,out1 = dialate_layer(shape_input,1,1,LAYER_WIDTH)
    final_res_l1,out2 = incr_pow2(out1,7)
    final_res_l2,out3 = incr_pow2(out2,8)
    final_res_l3,out4 = incr_pow2(out3,9)
    final_res_l4,out5 = incr_pow2(out4,9)
    final_result_concat = keras.layers.Concatenate(axis=2)(
        [final_res1] +
        final_res_l1 +
        final_res_l2 +
        final_res_l3 +
        final_res_l4
    )
    final_value = keras.layers.TimeDistributed(Dense(1))(final_result_concat)
    final_value_shape = Reshape((BLOCK_SIZE,))(final_value)

    model = Model(inputs=input, outputs=final_value_shape)

def stupid_loss_fn(input_batch,global_vector_batch):
    input_block = tf.concat([input_batch[:,:-1],global_vector_batch],axis=1)

    hidden_fn_1 = DenseLayer("a",int(input_block.shape[1]),HIDDEN_SIZE_1)
    hidden_fn_2 = DenseLayer("b",HIDDEN_SIZE_1,HIDDEN_SIZE_2)
    output_fn = DenseLayer("c",HIDDEN_SIZE_2,1)

    hidden1 = tf.tanh(hidden_fn_1.calc_output(input_block))
    hidden2 = tf.tanh(hidden_fn_2.calc_output(hidden1))

    output = tf.sigmoid(output_fn.calc_output(hidden2))

    loss = tf.square(output - input_batch[:,-1])
    return loss
