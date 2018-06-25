import tensorflow as tf
from WeightBias import DenseLayer

LAYER_WIDTH=16
BATCH_SIZE=1
HIDDEN_SIZE=64

QUANTIZATION_CHANNELS=64

BLOCK_SIZE = 3000

SONG_VECTOR_SIZE = 24

USE_SCALAR_INPUT = False

USE_GPU = False

STANDARD_SAVE_REPO = "../relu_repo_results/"

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)

def quantitize_audio_to_1hot(audio,quantization_channels):
    quant = mu_law_encode(audio, quantization_channels)
    vecs = tf.one_hot(quant,quantization_channels)
    return vecs

def shift_x(x,shift_ammount):
    val = tf.manip.roll(x,shift=shift_ammount,axis=1)
    return val

def dialate_layer(input, glob_vector, dialate_length):
    glob_vec_shape = tf.reshape(glob_vector,(BATCH_SIZE,1,SONG_VECTOR_SIZE))
    padded_glob_vec = tf.tile(glob_vec_shape,(1,BLOCK_SIZE,1))
    rolled_layer = shift_x(input,dialate_length)
    concat_input = tf.concat([input,rolled_layer,padded_glob_vec],axis=2)
    in_layer_shape = LAYER_WIDTH*2+SONG_VECTOR_SIZE
    relu_activ = tf.nn.relu(DenseLayer('relu',in_layer_shape,LAYER_WIDTH).calc_output(concat_input))
    # NOTE: wavenet puts another matrix multiplication here1
    feed_forward = tf.add(relu_activ,input)
    return relu_activ,feed_forward


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
    if USE_SCALAR_INPUT:
        shape_input = tf.reshape(input_batch,(BATCH_SIZE,BLOCK_SIZE,1))
        final_input_val = tf.tile(shape_input,(1,1,LAYER_WIDTH))
    else:
        quant_shape = quantitize_audio_to_1hot(input_batch,QUANTIZATION_CHANNELS)
        final_input_val = DenseLayer('linear',QUANTIZATION_CHANNELS,LAYER_WIDTH).calc_output(quant_shape)# no activation function because it does not need to learn anything complicated here

    MAX_COVER_SIZE = 2*(2*2**9 + 2**8 + 2**7) + 8
    final_res_l1,out2 = incr_pow2(final_input_val,glob_vecs,7)
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
    final_hidden_layer =  tf.tanh(DenseLayer('relu',final_res_size,HIDDEN_SIZE).calc_output(final_result_concat))

    final_scalar_value = tf.tanh(DenseLayer('tanh',HIDDEN_SIZE,1).calc_output(final_hidden_layer))
    final_scalar_value_flat = tf.reshape(final_scalar_value,(BATCH_SIZE,BLOCK_SIZE))

    final_value = final_scalar_value_flat[:,MAX_COVER_SIZE+1:]
    input_compare = input_batch[:,MAX_COVER_SIZE:-1]

    loss = tf.reduce_mean(tf.abs(final_value-input_compare))
    return loss
