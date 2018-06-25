import tensorflow as tf
from WeightBias import DenseLayer

LAYER_WIDTH=16
BATCH_SIZE=4
HIDDEN_SIZE=64

BLOCK_SIZE = 3000

SONG_VECTOR_SIZE = 16

BATCH_SIZE = 4




def shift_x(x,shift_ammount):
    # the 0 axis here is the batch axis, so the 1 axis
    # is the 0 axis to kerasa
    tensorflow_0_axis = 1
    val = tf.manip.roll(x,shift=shift_ammount,axis=tensorflow_0_axis)
    return val

def dialate_layer(input, glob_vector, dialate_length):
    glob_vec_tranform = DenseLayer('glob',SONG_VECTOR_SIZE,LAYER_WIDTH).calc_output(glob_vector)
    glob_vec_shape = tf.reshape(glob_vec_tranform,(BATCH_SIZE,1,LAYER_WIDTH))
    rolled_layer = shift_x(input,-dialate_length)
    concat_input = tf.concat([input,rolled_layer],axis=2)
    glob_added = tf.add(concat_input,tf.tile(glob_vec_shape,(1,1,2)))
    sigmoid_part = tf.sigmoid(DenseLayer('sig',LAYER_WIDTH*2,LAYER_WIDTH).calc_output(glob_added))
    tanh_part = tf.tanh(DenseLayer('tanh',LAYER_WIDTH*2,LAYER_WIDTH).calc_output(glob_added))
    res = tf.multiply(sigmoid_part,tanh_part)
    # NOTE: wavenet puts another matrix multiplication here1
    feed_forward = tf.add(res,input)
    return res,feed_forward


def incr_pow2(input,glob_vecs, highest_pow2):
    final_list = []
    cur_input = input
    for i in range(0,highest_pow2+1):
        skip_len = 2**i
        final_res2,out2 = dialate_layer(cur_input,glob_vecs,skip_len)
        cur_input = out2
        final_list.append(final_res2)
    return final_list, cur_input


def wavenet_loss(input_batch,glob_vecs):
    shape_input = tf.reshape(input_batch,(BATCH_SIZE,BLOCK_SIZE,1))
    padded_shape_input = tf.tile(shape_input,(1,1,LAYER_WIDTH))
    MAX_COVER_SIZE = 2*(2*2**9 + 2**8 + 2**7) + 8
    final_res_l1,out2 = incr_pow2(padded_shape_input,glob_vecs,7)
    final_res_l2,out3 = incr_pow2(out2,glob_vecs,8)
    final_res_l3,out4 = incr_pow2(out3,glob_vecs,9)
    final_res_l4,out5 = incr_pow2(out4,glob_vecs,9)
    final_res_list = (
        final_res_l1 +
        final_res_l2 +
        final_res_l3 +
        final_res_l4
    )
    final_result_concat = tf.concat(final_res_list,axis=2)

    final_res_size = LAYER_WIDTH * len(final_res_list)
    final_hidden_layer =  tf.tanh(DenseLayer('tanh',final_res_size,HIDDEN_SIZE).calc_output(final_result_concat))

    final_scalar_value = tf.tanh(DenseLayer('tanh',HIDDEN_SIZE,1).calc_output(final_hidden_layer))
    final_scalar_value_flat = tf.reshape(final_scalar_value,(BATCH_SIZE,BLOCK_SIZE))

    final_value = final_scalar_value_flat[:,MAX_COVER_SIZE+1:]
    input_compare = input_batch[:,MAX_COVER_SIZE:-1]

    loss = tf.reduce_mean(tf.abs(final_value-input_compare))
    return loss

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
