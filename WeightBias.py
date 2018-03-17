import theano
import numpy as np
import theano.tensor as T

class WeightBias:
    def __init__(self,name,in_len,out_len,mean_val=0):
        self.name = name

        rand_weight_vals = np.random.randn(in_len*out_len).astype('float32')/in_len**(0.5**0.5) #+ mean_val / (in_len)
        rand_weight = np.reshape(rand_weight_vals,(out_len,in_len))
        self.W = theano.shared(rand_weight,name=self.weight_name())

        bias_init = np.zeros(out_len,dtype='float32') + mean_val
        self.b = theano.shared(bias_init,name=self.bias_name())#allows nice addition

    def calc_output(self,in_tens):
        if in_tens.ndim == 1:
            prod = T.dot(self.W,in_tens)
            return T.add(prod,self.b)
        elif in_tens.ndim == 2:
            #batched inputs
            prod = T.dot(self.W,in_tens)
            return T.add(self.b[:,None],prod)
    def bias_name(self):
        return self.name+"b"
    def weight_name(self):
        return self.name+"W"
    def wb_list(self):
        return [self.W,self.b]
    def update(self,error,train_update_const):
        Wg,bg = T.grad(error,self.wb_list())

        c = train_update_const
        return [(self.W,self.W-Wg*c),(self.b,self.b-bg*c)]

class NP_WeightBias:
    def __init__(self,theano_weight_bias):
        self.W = theano_weight_bias.W.get_value()
        self.b = theano_weight_bias.b.get_value()
    def calc_output(self,invec):
        return np.dot(self.W,invec) + self.b
    calc_output_batched = calc_output
