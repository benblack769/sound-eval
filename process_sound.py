import numpy as np
import soundfile as sf
import theano
import theano.tensor as T
import plot_utility
import matplotlib.pyplot as plt
from shared_save import RememberSharedVals

from WeightBias import WeightBias
theano.config.optimizer="fast_compile"
NUM_LOOK_BACK = 20
plotutil = plot_utility.PlotHolder("basic_testa")
shared_value_saver = RememberSharedVals('basic_weights')

def square(x):
    return x * x

def get_pred_train_fns(inlen, outlen, hiddenlen, train_update_const):
    inputvec = T.vector('invec', dtype=theano.config.floatX)
    expectedvec = T.vector('expected', dtype=theano.config.floatX)

    hiddenfn = WeightBias("hidden",inlen, hiddenlen)
    outfn = WeightBias("out",hiddenlen, outlen)

    shared_value_saver.add_shared_vals(hiddenfn.wb_list())
    shared_value_saver.add_shared_vals(outfn.wb_list())

    def logistic(x):
        one = np.float32(1.0)
        return one / (one+T.exp(-x))

    hidvec = logistic(hiddenfn.calc_output(inputvec))
    outvec = logistic(outfn.calc_output(hidvec))

    actual = outvec
    diff = square(expectedvec - actual)
    error = diff.sum()

    outdiff = outfn.update(error,np.float32(train_update_const))
    hiddiff = hiddenfn.update(error,np.float32(train_update_const))

    hidbias_plot = plotutil.add_plot("hidbias",hiddenfn.b,1000)
    outbias_plot = plotutil.add_plot("outbias",outfn.b,1000)
    outbias_plot = plotutil.add_plot("error",error,100)

    predict = theano.function(
            inputs=[inputvec],
            outputs=[outvec]
        )

    train = theano.function(
            inputs=[inputvec,expectedvec],
            outputs=plotutil.append_plot_outputs([]),
            updates=hiddiff+outdiff,
        )
    saved_train_fn = shared_value_saver.share_save_fn(train)
    plotted_train_fn = plotutil.get_plot_update_fn(saved_train_fn)
    return predict, plotted_train_fn

def train_on_data(sig, samplerate, my_train_fn):
    for start_loc in range(len(sig) - NUM_LOOK_BACK - 1):
        inp_mat = sig[start_loc:start_loc+NUM_LOOK_BACK]
        inp_vec = np.ndarray.flatten(inp_mat)
        ex = sig[start_loc+NUM_LOOK_BACK]
        my_train_fn(inp_vec,ex)

def rand_starter(size):
    return np.random.randn(size).astype('float32')/3


def predict(sig, samplerate, my_predict_fn):
    res_list = [rand_starter(2) for x in range(NUM_LOOK_BACK)]
    #print(res_list)
    for i in range(1000):
        #print(res_list)
        inp = np.concatenate(res_list[-3:])
        output = my_predict_fn(inp)[0]
        res_list.append(output)
        print(output)




sig, samplerate = sf.read('output.wav')
sig = sig.astype(np.float32)
num_channels = sig.shape[1]
pred_fn, train_fn = get_pred_train_fns(num_channels*NUM_LOOK_BACK,num_channels,30,0.1)
train_on_data(sig, samplerate, train_fn)
#predict(sig, samplerate, pred_fn)
