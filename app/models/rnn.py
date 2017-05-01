# -*- coding: utf-8 -*-
"""
rnn

model built atop tensorflow for constructing Recurrent Neural Network (RNN) and
Long Short-Term Memory (LSTM) architecture
"""

import tensorflow as tf


class RecurrentNeuralNet:

    def __init__(self, state_size, num_classes):

        self._state_size = state_size
        self._n_class = num_classes

    @property
    def x(self):
        """feature set"""
        return tf.placeholder(
                            dtype=tf.float32,
                            shape=(-1, ) + self.shape,
                            name='feature')

    @property
    def y_(self):
        """true label, in one hot format"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._n_class),
                              name='multi-label')

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        """LSTM implementation by Hochreiter & Schmidhuber (1997)."""
        return tf.contrib.rnn.LSTMCell(num_units=self.nodes,
                                       activation=tf.tanh,
                                       use_peepholes=False)

    def lstm(self):
        """dynamic_rnn pad sequential input of different sizes."""
        outputs, state = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                           inputs=self.x,
                                           dtype=tf.float32)
        return outputs, state
