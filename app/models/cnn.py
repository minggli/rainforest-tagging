# -*- coding: utf-8 -*-
"""
cnn

template built atop tensorflow for constructing Convolutional Neural Network
"""

import tensorflow as tf


class ConvolutionalNeuralNetwork:

    def __init__(self, shape, num_classes):
        """shape: [n_samples, channels, n_features]"""
        self._shape = shape
        self._n_class = num_classes

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        """
        core operation that convolves through image data and extract features
        input: takes a 4-D shaped tensor e.g. (-1, 90, 160, 3)
        receptive field (filter): filter size and number of output channels are
            inferred from weight hyperparams.
        receptive field moves by 1 pixel at a time during convolution.
        Zero Padding algorthm appies to keep output size the same
            e.g. 3x3xn filter with 1 zero-padding, 5x5 2, 7x7 3 etc.
        """
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

    @staticmethod
    def max_pool(x):
        """max pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.max_pool(value=x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def non_linearity(activation):
        if activation == 'sigmoid':
            return tf.nn.sigmoid
        elif activation == 'relu':
            return tf.nn.relu
        elif activation == 'tanh':
            return tf.tanh

    @property
    def x(self):
        """feature set"""
        return tf.placeholder(dtype=tf.float32,
                              # transform 3D shape to 4D to include batch size
                              shape=(None, ) + self._shape,
                              name='feature')

    @property
    def y_(self):
        """true label, in one hot format"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._n_class),
                              name='label')

    # @property
    # def keep_prob(self):
    #     return tf.placeholder(dtype=tf.float32, name='keepprob')

    def add_conv_layer(self, input_layer, hyperparams, func='relu'):
        """Convolution Layer with hyperparamters and activation_func"""
        W = self.weight_variable(shape=hyperparams[0])
        b = self.bias_variable(shape=hyperparams[1])

        hypothesis_conv = self.non_linearity(func)(
                          self.conv2d(input_layer, W) + b)
        return hypothesis_conv

    def add_pooling_layer(self, input_layer):
        """max pooling layer to reduce overfitting"""
        hypothesis_pool = self.max_pool(input_layer)
        return hypothesis_pool

    def add_dense_layer(self, input_layer, hyperparams, func='relu'):
        """Densely Connected Layer with hyperparamters and activation_func"""
        W = self.weight_variable(shape=hyperparams[0])
        b = self.bias_variable(shape=hyperparams[1])

        flat_x = tf.reshape(input_layer, hyperparams[2])
        hypothesis = self.non_linearity(func)(tf.matmul(flat_x, W) + b)
        return hypothesis

    def add_drop_out_layer(self, input_layer, keep_prob):
        """drop out layer to reduce overfitting"""
        hypothesis_drop = tf.nn.dropout(input_layer, keep_prob)
        return hypothesis_drop

    def add_read_out_layer(self, input_layer):
        """read out layer with output shape of [batch_size, num_classes]
        to feed into softmax"""
        input_layer_m = int(input_layer.get_shape()[1])
        W = self.weight_variable(shape=[input_layer_m, self._n_class])
        b = self.bias_variable(shape=[self._n_class])

        logits = tf.matmul(input_layer, W) + b
        return logits
