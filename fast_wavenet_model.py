'''
copied over from  https://github.com/tomlepaine/fast-wavenet

under GNU liscence for authors.
'''
import tensorflow as tf

from fast_wavenet_layers import *

class Model(object):
    def __init__(self,
                 num_time_samples,
                 batch_size=32,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 gpu_fraction=1.0):

        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.gpu_fraction = gpu_fraction

        inputs = tf.placeholder(tf.float32,
                                shape=(None, num_time_samples, num_channels))
        targets = tf.placeholder(tf.int32, shape=(None, num_time_samples))

        h = inputs
        hs = []
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                name = 'b{}-l{}'.format(b, i)
                h = dilated_conv1d(h, num_hidden, rate=rate, name=name)
                hs.append(h)

        outputs = conv1d(h,
                         num_classes,
                         filter_width=1,
                         gain=1.0,
                         activation=None,
                         bias=True)

        costs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            outputs, targets)
        cost = tf.reduce_mean(costs)

        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        self.inputs = inputs
        self.targets = targets
        self.outputs = outputs
        self.costs = costs
        self.cost = cost
        self.train_step = train_step

    def train_step(self, sess, inputs, targets):
        feed_dict = {self.inputs: inputs, self.targets: targets}
        cost, _ = sess.run(
            [self.cost, self.train_step],
            feed_dict=feed_dict)
        return cost
