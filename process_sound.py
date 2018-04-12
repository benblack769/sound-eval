import numpy as np
import soundfile as sf
import theano
import theano.tensor as T
import plot_utility
import matplotlib.pyplot as plt
from shared_save import RememberSharedVals
from WeightBias import WeightBias
from opts import RMSpropOpt

#theano.config.optimizer="fast_compile"

NUM_LOOK_BACK = 20
SKIP4_TIER_SIZE = 21
SKIP2_TIER_SIZE = 24
SKIP1_TIER_SIZE = 27
RMS_LEARN_RATE = 0.001
HIDDEN_SIZE = SKIP1_TIER_SIZE + SKIP2_TIER_SIZE + SKIP4_TIER_SIZE
BATCH_SIZE = 32
plotutil = plot_utility.PlotHolder("basic_test")
shared_value_saver = RememberSharedVals('basic_weights')

def square(x):
    return x * x

def get_pred_train_fns(inlen, outlen, train_update_const):
    inputvec = T.matrix('invec', dtype=theano.config.floatX)
    expectedvec = T.vector('expected', dtype=theano.config.floatX)

    act_in_len = HIDDEN_SIZE + inlen
    hiddenS4 = WeightBias("hiddenS4", act_in_len, SKIP4_TIER_SIZE)
    hiddenS2 = WeightBias("hiddenS2", act_in_len, SKIP2_TIER_SIZE)
    hiddenS1 = WeightBias("hiddenS1", act_in_len, SKIP1_TIER_SIZE)
    outfn = WeightBias("out",HIDDEN_SIZE, outlen)
    total_wb_list = hiddenS4.wb_list()+hiddenS2.wb_list()+hiddenS1.wb_list()+outfn.wb_list()
    optimizer = RMSpropOpt(total_wb_list,RMS_LEARN_RATE)

    plotutil.add_plot("skip4_weights", hiddenS4.W ,1000)
    plotutil.add_plot("skip2_weights", hiddenS2.W ,1000)
    plotutil.add_plot("skip1_weights", hiddenS1.W ,1000)
    plotutil.add_plot("out_weights", outfn.W ,1000)

    shared_value_saver.add_shared_vals(hiddenS4.wb_list())
    shared_value_saver.add_shared_vals(hiddenS2.wb_list())
    shared_value_saver.add_shared_vals(hiddenS1.wb_list())
    shared_value_saver.add_shared_vals(outfn.wb_list())
    shared_value_saver.add_shared_vals(optimizer.get_shared_states())

    def logistic(x):
        one = np.float32(1.0)
        return one / (one+T.exp(-x))

    skip4_intermed = T.zeros_like(hiddenS4.b[:SKIP4_TIER_SIZE])
    skip2_intermed = T.zeros_like(hiddenS2.b[:SKIP2_TIER_SIZE])
    skip1_intermed = T.zeros_like(hiddenS1.b[:SKIP1_TIER_SIZE])
    for i in range(NUM_LOOK_BACK):
        total_in_vec = T.concatenate([inputvec[i], skip1_intermed, skip2_intermed, skip4_intermed])
        skip1_intermed = logistic(hiddenS1.calc_output(total_in_vec))
        if i % 2 == 1:
            skip2_intermed = logistic(hiddenS2.calc_output(total_in_vec))
        if i % 4 == 3:
            skip4_intermed = logistic(hiddenS4.calc_output(total_in_vec))

    total_out_in_vec = T.concatenate([skip1_intermed, skip2_intermed, skip4_intermed])
    out_val = logistic(outfn.calc_output(total_out_in_vec))

    actual = out_val
    diff = square(expectedvec - actual)
    error = diff.sum()

    plotutil.add_plot("error",error,100)

    all_updates = optimizer.updates(error,total_wb_list)
    #outdiff = outfn.update(error,np.float32(train_update_const))
    #hiddenS4_u = hiddenS4.update(error,np.float32(train_update_const))
    #hiddenS2_u = hiddenS2.update(error,np.float32(train_update_const))
    #hiddenS1_u = hiddenS1.update(error,np.float32(train_update_const))

    #hidbias_plot = plotutil.add_plot("hidbias",hiddenfn.b,100)
    #outbias_plot = plotutil.add_plot("outbias",outfn.b,100)

    predict = theano.function(
            inputs=[inputvec],
            outputs=[out_val]
        )

    train = theano.function(
            inputs=[inputvec,expectedvec],
            outputs=plotutil.append_plot_outputs([]),
            updates=all_updates,
        )
    saved_train_fn = shared_value_saver.share_save_fn(train)
    plotted_train_fn = plotutil.get_plot_update_fn(saved_train_fn)
    return predict, plotted_train_fn

def get_input_batch(start_loc):
    for loc in range(start_loc, start_loc + BATCH_SIZE):
        inp_mat = sig[loc:loc+NUM_LOOK_BACK]
        inp_vec = np.ndarray.flatten(inp_mat)

def train_on_data(sig, samplerate, my_train_fn):
    for start_loc in range(len(sig) - NUM_LOOK_BACK - 1 - BATCH_SIZE - 2):
        inp_mat = sig[start_loc:start_loc+NUM_LOOK_BACK]
        #inp_vec = np.ndarray.flatten(inp_mat)
        ex = sig[start_loc+NUM_LOOK_BACK]
        my_train_fn(inp_mat,ex)

def rand_starter(size):
    return np.random.randn(size).astype('float32')/3


def predict_next(sig, samplerate, my_predict_fn):
    for start_loc in range(len(sig) - NUM_LOOK_BACK - 1 - BATCH_SIZE - 2):
        inp_mat = sig[start_loc:start_loc+NUM_LOOK_BACK]
        #inp_vec = np.ndarray.flatten(inp_mat)
        output = my_predict_fn(inp_mat)[0]
        ex = sig[start_loc+NUM_LOOK_BACK]
        diff = output - ex
        tot_diff = np.sum(np.absolute(diff))
        #print("out: {}".format(output))
        #print("ex: {}".format(ex))
        #print("tot: {}".format(tot_diff))



sig, samplerate = sf.read('output.wav')
sig = sig.astype(np.float32)
num_channels = sig.shape[1]
pred_fn, train_fn = get_pred_train_fns(num_channels,num_channels,0.1)
print("compiled fn\n\n\n!!!!!")
train_on_data(sig, samplerate, train_fn)
#predict_next(sig, samplerate, pred_fn)
