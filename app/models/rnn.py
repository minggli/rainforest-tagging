# -*- coding: utf-8 -*-
"""
rnn

model built with tensorflow for constructing standard Recurrent Neural Network
(RNN) and Long Short-Term Memory (LSTM)

"""

import tensorflow as tf


class _BaseRNN(object):
    """base class recurrent neural network"""
    def __init__(self, step_size, state_size, num_classes):

        self._step_size = step_size
        self._state_size = state_size
        self._n_class = num_classes

    @property
    def x(self):
        """feature vector"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._step_size),
                              name='feature')

    @property
    def y_(self):
        """true label, in one hot format"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._n_class),
                              name='multilabel')

    @property
    def _zero_state(self):
        """zero state for initial state and initial output"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._state_size))

    def get_weight_variable(self, name):
        """create new or reuse weight variable by variable name."""
        init = tf.truncated_normal_initializer(stddev=.1)
        if 'W_hx' in name:
            return tf.get_variable(name=name,
                                   shape=[self._step_size, self._state_size],
                                   initializer=init)
        elif 'W_hh' in name:
            return tf.get_variable(name=name,
                                   shape=[self._state_size, self._state_size],
                                   initializer=init)
        else:
            raise RuntimeError('must specify hx or hh for rnn cell weights.'
                               'such an elegant hack :).')

    def get_bias_variable(self, name):
        """create new or reuse bias variable by variable name."""
        init = tf.constant_initializer(0.0)
        return tf.get_variable(name=name,
                               shape=[self._state_size],
                               initializer=init)

    def static_rnn(self):
        """whereas dynamic rnn is known to be preferred due to its
        flexibility, this static approach aims to lay foundation to dynamic rnn
        implementation."""
        rnn_inputs = tf.unstack(self.x, axis=1)
        output = state = tf.zeros_like(self._zero_state, name='initial_state')
        output_receiver = list()

        if self.__class__.__name__ == 'RNN':
            for rnn_input in rnn_inputs:
                output, state = self.__call__(rnn_input=rnn_input, state=state)
                output_receiver.append(output)
        elif self.__class__.__name__ == 'LSTM':
            for rnn_input in rnn_inputs:
                output, state = self.__call__(cell_input=rnn_input,
                                              cell_output=output,
                                              cell_state=state)
                output_receiver.append(output)
        else:
            raise Exception('_BaseRNN can not be called directly.')
        return output_receiver, state


class RNN(_BaseRNN):

    def __init__(self, step_size, state_size, num_classes):
        super(RNN, self).__init__(step_size, state_size, num_classes)

    def __call__(self, rnn_input, state):
        """RNN cell implementation to Colah's blog (2015)."""
        with tf.variable_scope('default_rnn_cell', reuse=None):
            W_hx = self.get_weight_variable(name='W_hx')
            W_hh = self.get_weight_variable(name='W_hh')
            b_h = self.get_bias_variable(name='b_h')

        output = tf.tanh(tf.matmul(rnn_input, W_hx) +
                         tf.matmul(state, W_hh) +
                         b_h)
        state = output
        return output, state


class LSTM(_BaseRNN):

    def __init__(self, step_size, state_size, num_classes):
        super(LSTM, self).__init__(step_size, state_size, num_classes)

    def __call__(self, cell_input, cell_output, cell_state):
        """LSTM cell implemented to Hochreiter & Schmidhuber (1997)"""
        with tf.variable_scope('default_lstm_cell', reuse=None):
            forget_W_hx = self.get_weight_variable(name='forget_W_hx')
            forget_W_hh = self.get_weight_variable(name='forget_W_hh')
            forget_b_h = self.get_bias_variable(name='forget_b_h')
            input_W_hx = self.get_weight_variable(name='input_W_hx')
            input_W_hh = self.get_weight_variable(name='input_W_hh')
            input_b_h = self.get_bias_variable(name='input_b_h')
            cell_state_W_hx = self.get_weight_variable(name='cell_state_W_hx')
            cell_state_W_hh = self.get_weight_variable(name='cell_state_W_hh')
            cell_state_b_h = self.get_bias_variable(name='cell_state_b_h')
            output_W_hx = self.get_weight_variable(name='output_W_hx')
            output_W_hh = self.get_weight_variable(name='output_W_hh')
            output_b_h = self.get_bias_variable(name='output_b_h')

        forget_gate = tf.sigmoid(
                        tf.matmul(cell_input, forget_W_hx) +
                        tf.matmul(cell_output, forget_W_hh) +
                        forget_b_h)
        input_gate = tf.sigmoid(
                        tf.matmul(cell_input, input_W_hx) +
                        tf.matmul(cell_output, input_W_hh) +
                        input_b_h)
        cell_state_delta = tf.tanh(
                        tf.matmul(cell_input, cell_state_W_hx) +
                        tf.matmul(cell_output, cell_state_W_hh) +
                        cell_state_b_h)

        # cell memory forgets old information and learns new information
        cell_state_t = forget_gate * cell_state + input_gate * cell_state_delta

        output_gate = tf.sigmoid(
                      tf.matmul(cell_input, output_W_hx) +
                      tf.matmul(cell_output, output_W_hh) +
                      output_b_h)

        output = output_gate * tf.tanh(cell_state_t)
        return output, cell_state_t
