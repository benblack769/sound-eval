import numpy as np
import theano
import theano.tensor as T

class RMSpropOpt:
    def __init__(self, update_shared, LEARNING_RATE, DECAY_RATE=0.98):
        self.LEARNING_RATE = np.float32(LEARNING_RATE)
        self.DECAY_RATE = np.float32(DECAY_RATE)
        self.grad_sqrd_mags = []
        for update_val in update_shared:
            start_vals = np.ones(update_val.get_value().shape,dtype=np.float32) * 4.0
            self.grad_sqrd_mags.append(theano.shared(start_vals,"grad_sqrd_mag"+update_val.name))

    def get_shared_states(self):
        return self.grad_sqrd_mags

    def updates(self,error,weight_biases):
        STABILIZNG_VAL = np.float32(0.0001**2)
        all_grads = T.grad(error,wrt=weight_biases)

        all_updates = []
        for grad_val, g_sqr_mag, wb in zip(all_grads, self.grad_sqrd_mags, weight_biases):
            grad_sqrd = grad_val * grad_val
            grad_sqrd_mag_update = self.DECAY_RATE * g_sqr_mag + (np.float32(1)-self.DECAY_RATE)*grad_sqrd

            wb_update_mag = self.LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)

            all_updates.append((wb, wb - wb_update_mag * grad_val))
            all_updates.append((g_sqr_mag, grad_sqrd_mag_update))

        return all_updates

class RMSpropSINGLEOpt:
    def __init__(self,LEARNING_RATE,DECAY_RATE=0.9):
        self.LEARNING_RATE = np.float32(LEARNING_RATE)
        self.DECAY_RATE = np.float32(DECAY_RATE)
        self.grad_sqrd_mag = theano.shared(np.float32(400),"grad_sqrd_mag")

    def get_shared_states(self):
        return [self.grad_sqrd_mag]

    def updates(self,error,weight_biases):
        STABILIZNG_VAL = np.float32(0.0001**2)
        all_grads = T.grad(error,wrt=weight_biases)

        gsqr = sum(T.sum(g*g) for g in all_grads)

        grad_sqrd_mag_update = self.DECAY_RATE * self.grad_sqrd_mag + (np.float32(1)-self.DECAY_RATE)*gsqr

        wb_update_mag = self.LEARNING_RATE / T.sqrt(grad_sqrd_mag_update + STABILIZNG_VAL)

        wb_update = [(wb,wb - wb_update_mag * grad) for wb,grad in zip(weight_biases,all_grads)]
        return wb_update + [(self.grad_sqrd_mag,grad_sqrd_mag_update)]
