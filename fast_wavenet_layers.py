'''
copied over from  https://github.com/tomlepaine/fast-wavenet

under GNU liscence for authors.
'''

import numpy as np
import tensorflow as tf
from WeightBias import DenseLayer

def dialated_conv(name,inputs,out_channels,dialation):
    '''
    input format: [batch_size,elements,in_channels]
    kernel_format: [2*in_channels]->[out_channels]
    '''
    assert dialation >= 1
    inputs_1 = inputs[:,dialation:]
    inputs_2 = inputs[:,:-dialation]
    result = tf.concat([inputs_1,inputs_2],axis=2)

    sig_kernel_matrix = DenseLayer(name,result.shape[2],out_channels)
    tanh_kernel_matrix = DenseLayer(name,result.shape[2],out_channels)
    kernel_matrix.calc_output(result)
