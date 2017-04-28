# -*- coding: utf-8 -*-
"""
rnn

model built atop tensorflow for constructing Recurrent Neural Network (RNN) and
Long Short-Term Memory (LSTM) architecture
"""

import tensorflow as tf


class RecurrentNeuralNet:

    def __init__(self, cell_type, num_nodes):
        self.type = cell_type
        self.nodes = num_nodes

    @property
    def x(self):
        """feature set"""
        return tf.reshape(tf.placeholder(
            dtype=tf.float32,
            shape=self.flattened_shape,
            name='feature'
        ),
            # transform 3D shape to 4D
            (-1, ) + self.shape
        )

    @property
    def y_(self):
        """true label, in one hot format"""
        return tf.placeholder(dtype=tf.float32, shape=[None, 8], name='label')

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def cell(self):
        """vanila RNN cell or LSTM cell (Hochreiter & Schmidhuber 1997)
        with forget, input, and output gates. """

        if self.type == 'RNN':
            return tf.contrib.rnn.BasicRNNCell(
                                            num_units=self.nodes,
                                            activation=tf.tanh)
        elif self.type == 'LSTM':
            return tf.contrib.rnn.LSTMCell(num_units=self.nodes,
                                           activation=tf.tanh,
                                           use_peepholes=False)

    def _unrolled_rnn(self):
        """dynamic_rnn padd sequential input of different sizes."""
        outputs, state = tf.nn.dynamic_rnn(cell=self.cell,
                                           inputs=self.x,
                                           dtype=tf.float32)
        return outputs, state
