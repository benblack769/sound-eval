import numpy as np
import soundfile as sf
import theano
import theano.tensor as T
import plot_utility
import matplotlib.pyplot as plt
from shared_save import RememberSharedVals
from WeightBias import WeightBias

#theano.config.optimizer="fast_compile"

NUM_LOOK_BACK = 20
SLOW_TIER_SKIP = 4
SLOW_TIER_SIZE = 35
FAST_TIER_SIZE = 28
BATCH_SIZE = 32
plotutil = plot_utility.PlotHolder("basic_test")
shared_value_saver = RememberSharedVals('basic_weights')

def square(x):
    return x * x

def get_pred_train_fns(inlen, outlen, train_update_const):
    inputvec = T.matrix('invec', dtype=theano.config.floatX)
    expectedvec = T.vector('expected', dtype=theano.config.floatX)

    act_in_len = SLOW_TIER_SIZE + FAST_TIER_SIZE + inlen
    hiddenSFS = WeightBias("hiddenSFS", act_in_len, SLOW_TIER_SIZE)
    hiddenSFF = WeightBias("hiddenSFF", act_in_len, FAST_TIER_SIZE)
    outfn = WeightBias("out",SLOW_TIER_SIZE + FAST_TIER_SIZE, outlen)

    shared_value_saver.add_shared_vals(hiddenSFS.wb_list())
    shared_value_saver.add_shared_vals(hiddenSFF.wb_list())
    shared_value_saver.add_shared_vals(outfn.wb_list())

    def logistic(x):
        one = np.float32(1.0)
        return one / (one+T.exp(-x))

    slow_intermed = T.zeros_like(hiddenSFS.b[:SLOW_TIER_SIZE])
    fast_intermed = T.zeros_like(hiddenSFF.b[:FAST_TIER_SIZE])
    for i in range(NUM_LOOK_BACK):
        total_in_vec = T.concatenate([inputvec[i], slow_intermed, fast_intermed])
        fast_intermed = logistic(hiddenSFF.calc_output(total_in_vec))
        if i % SLOW_TIER_SKIP == SLOW_TIER_SKIP - 1:
            slow_intermed = logistic(hiddenSFS.calc_output(total_in_vec))


    total_out_in_vec = T.concatenate([slow_intermed, fast_intermed])
    out_val = logistic(outfn.calc_output(total_out_in_vec))

    actual = out_val
    diff = square(expectedvec - actual)
    error = diff.sum()
    plotutil.add_plot("error",error,100)

    outdiff = outfn.update(error,np.float32(train_update_const))
    hiddiffSFS = hiddenSFS.update(error,np.float32(train_update_const))
    hiddiffSFF = hiddenSFF.update(error,np.float32(train_update_const))

    #hidbias_plot = plotutil.add_plot("hidbias",hiddenfn.b,100)
    #outbias_plot = plotutil.add_plot("outbias",outfn.b,100)

    predict = theano.function(
            inputs=[inputvec],
            outputs=[out_val]
        )

    train = theano.function(
            inputs=[inputvec,expectedvec],
            outputs=plotutil.append_plot_outputs([]),
            updates=hiddiffSFS+hiddiffSFF+outdiff,
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
        print("out: {}".format(output))
        print("ex: {}".format(ex))
        print("tot: {}".format(tot_diff))



sig, samplerate = sf.read('output.wav')
sig = sig.astype(np.float32)
num_channels = sig.shape[1]
pred_fn, train_fn = get_pred_train_fns(num_channels,num_channels,0.1)
print("compiled fn\n\n\n!!!!!")
#train_on_data(sig, samplerate, train_fn)
predict_next(sig, samplerate, pred_fn)
