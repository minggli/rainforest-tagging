# -*- coding: utf-8 -*-
"""
rnn

model built atop tensorflow for constructing Recurrent Neural Network (RNN) and
Long Short-Term Memory (LSTM) architecture
"""

import tensorflow as tf


class RecurrentNeuralNetwork:

    def __init__(self, state_size, num_classes, cell_type):

        self._cell_type = None
        self._state_size = state_size
        self._n_class = num_classes
        self.__cell_type__ = cell_type

    @property
    def x(self):
        """feature vector"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(-1, ) + self.shape,
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

    @property
    def __cell_type__(self):
        return self._cell_type

    @__cell_type__.setter
    def __cell_type__(self, value):
        try:
            if value.lower() in ['rnn', 'lstm']:
                self._cell_type = value.lower()
            else:
                raise RuntimeError
        except (AttributeError, RuntimeError) as e:
            raise Exception('Unknown cell type {0}, must be either'
                            '"rnn" or "lstm"'.format(value))

    def get_weight_variable(self, name):
        """create new or reuse weight variable by variable name."""
        init = tf.truncated_normal_initializer(stddev=.1)
        if 'W_hx' in name:
            return tf.get_variable(name=name,
                                   shape=[self._n_class, self._state_size],
                                   initializer=init)
        elif 'W_hh' in name:
            return tf.get_variable(name=name,
                                   shape=[self._n_class, self._state_size],
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

    def _rnn_cell(self, rnn_input, state):
        """RNN implementation to Colah's blog (2015)."""
        with tf.variable_scope('default_rnn_cell', reuse=None):
            W_hx = self.get_weight_variable(name='W_hx')
            W_hh = self.get_weight_variable(name='W_hh')
            b_h = self.get_bias_variable(name='b_h')

        output = tf.tanh(tf.matmul(rnn_input, W_hx) + tf.matmul(state, W_hh) +
                         b_h)
        state = output
        return output, state

    def _lstm_cell(self, cell_input, cell_output, cell_state):
        """implementation of LSTM to Hochreiter & Schmidhuber (1997)"""
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
        cell_state_t = \
            forget_gate * cell_state + input_gate * cell_state_delta

        output_gate = tf.sigmoid(
                      tf.matmul(cell_input, output_W_hx) +
                      tf.matmul(cell_output, output_W_hh) +
                      output_b_h
                      )
        output = output_gate * tf.tanh(cell_state_t)
        return output, cell_state_t

    def static_rnn(self):
        """whereas dynamic rnn is known to be preferred due to its
        flexibility, this static approach aims to lay foundation to dynamic rnn
        implementation."""
        rnn_inputs = tf.unstack(self.x, axis=1)
        output = state = tf.zeros_like(self._zero_state, name='initial_state')
        output_receiver = list()

        if self._cell_type == 'rnn':
            for rnn_input in rnn_inputs:
                output, state = self._rnn_cell(rnn_input=rnn_input,
                                               state=state)
                output_receiver.append(output)
        elif self._cell_type == 'lstm':
            for rnn_input in rnn_inputs:
                output, state = self._lstm_cell(cell_input=rnn_input,
                                                cell_output=output,
                                                cell_state=state)
                output_receiver.append(output)

        return output_receiver, state

    def __lstm_cell(self):
        """tensorflow's implementation to Hochreiter & Schmidhuber (1997)."""
        return tf.contrib.rnn.LSTMCell(num_units=self.nodes,
                                       activation=tf.tanh,
                                       use_peepholes=False)

    def __lstm(self):
        """dynamic_rnn pad varying time step sizes."""
        outputs, state = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                           inputs=self.x,
                                           dtype=tf.float32)
        return outputs, state
