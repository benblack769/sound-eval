import tensorflow as tf
import numpy as np

class DenseLayer:
    def __init__(self,name,in_len,out_len,mean_val=0):
        self.name = name

        rand_weight_vals = np.random.randn(out_len*in_len).astype('float32')/in_len**(0.5**0.5) #+ mean_val / (in_len)
        rand_weight = np.reshape(rand_weight_vals,(in_len,out_len))
        self.W = tf.Variable(rand_weight,name=self.weight_name())

        bias_init = np.random.randn(out_len).astype('float32')#np.zeros(out_len,dtype='float32') + mean_val
        self.b = tf.Variable(bias_init,name=self.bias_name())#allows nice addition

    def calc_output(self,in_tens):
        if(len(in_tens.shape) == 3):
            calc_tens = tf.reshape(in_tens,(in_tens.shape[0]*in_tens.shape[1],in_tens.shape[2]))
        else:
            calc_tens = in_tens
        prod = tf.matmul(calc_tens,self.W)
        res = tf.add(prod,self.b)

        if(len(in_tens.shape) == 3):
            res = tf.reshape(res,(in_tens.shape[0],in_tens.shape[1],res.shape[1]))
        return res

    def bias_name(self):
        return self.name+"b"

    def weight_name(self):
        return self.name+"W"

    def wb_list(self):
        return [self.W,self.b]

    def update(self,error,train_update_const):
        Wg,bg = tf.grad(error,self.wb_list())

        c = train_update_const
        return [(self.W,self.W-Wg*c),(self.b,self.b-bg*c)]

class Conv1dLayer(DenseLayer):
    def __init__(self):
        pass
    def calc_output(self,in_tens):
        prod = T.matmul(self.W,in_tens)
        return T.add(prod,self.b)




class NP_WeightBias:
    def __init__(self,theano_weight_bias):
        self.W = theano_weight_bias.W.get_value()
        self.b = theano_weight_bias.b.get_value()
    def calc_output(self,invec):
        return np.dot(self.W,invec) + self.b
    calc_output_batched = calc_output
