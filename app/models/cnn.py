# -*- coding: utf-8 -*-
"""
cnn

template atop tensorflow for buliding Convolutional Neural Network
"""
import operator
import functools
import tensorflow as tf


class ConvolutionalNeuralNet:

    def __init__(self, shape):
        """shape: [n_samples, channels, n_features]"""
        self.shape = shape
        self.flattened_shape = (
                                None,
                                shape[2],
                                functools.reduce(operator.mul, shape[:2], 1)
                                )

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
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool(x):
        """max pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def non_linearity(activation_func):
        if activation_func == 'sigmoid':
            return tf.nn.sigmoid
        elif activation_func == 'relu':
            return tf.nn.relu

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
    def _y(self):
        """true label, in one hot format"""
        return tf.placeholder(dtype=tf.float32, shape=[None, 8], name='label')

    # @property
    # def keep_prob(self):
    #     return tf.placeholder(dtype=tf.float32, name='keepprob')

    def add_conv_layer(self, input_layer, hyperparams, func='relu'):
        """Convolution Layer with hyperparamters and activation_func"""
        W = self.__class__.weight_variable(shape=hyperparams[0])
        b = self.__class__.bias_variable(shape=hyperparams[1])

        hypothesis_conv = self.__class__.non_linearity(func)(
            self.__class__.conv2d(input_layer, W) + b)
        return hypothesis_conv

    def add_pooling_layer(self, input_layer):
        """max pooling layer to reduce overfitting"""
        hypothesis_pool = self.__class__.max_pool(input_layer)
        return hypothesis_pool

    def add_dense_layer(self, input_layer, hyperparams, func='relu'):
        """Densely Connected Layer with hyperparamters and activation_func"""
        W = self.__class__.weight_variable(shape=hyperparams[0])
        b = self.__class__.bias_variable(shape=hyperparams[1])

        flat_x = tf.reshape(input_layer, hyperparams[2])
        hypothesis = \
            self.__class__.non_linearity(func)(tf.matmul(flat_x, W) + b)
        return hypothesis

    def add_drop_out_layer(self, input_layer, keep_prob):
        """drop out layer to reduce overfitting"""
        # keep_prob = self.keep_prob
        hypothesis_drop = tf.nn.dropout(input_layer, keep_prob)
        return hypothesis_drop

    def add_read_out_layer(self, input_layer, hyperparams):
        """read out layer"""
        W = self.__class__.weight_variable(shape=hyperparams[0])
        b = self.__class__.bias_variable(shape=hyperparams[1])

        logits = tf.matmul(input_layer, W) + b
        return logits
