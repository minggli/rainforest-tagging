# -*- coding: utf-8 -*-
"""
cnn

template built on top of tensorflow for constructing Convolutional Neural
Network
"""

import tensorflow as tf


class ConvolutionalNeuralNetwork:

    def __init__(self, shape, num_classes):
        """shape: [n_samples, channels, n_features]"""
        self._shape = shape
        self._n_class = num_classes

        self.is_train = self._is_train
        self.keep_prob = self._keep_prob

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name='weight')

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name='bias')

    @staticmethod
    def _conv2d(x, W):
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
    def _max_pool(x):
        """max pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.max_pool(value=x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def _nonlinearity(activation):
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
        """ground truth, in one-hot format"""
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._n_class),
                              name='label')

    @property
    def _keep_prob(self):
        """the probability constant to keep output from previous layer."""
        return tf.placeholder(dtype=tf.float32,
                              name='keep_rate')

    @property
    def _is_train(self):
        """indicates if network is under training mode."""
        return tf.placeholder(dtype=tf.bool,
                              name='is_train')

    def _batch_normalize(self, input_layer):
        """batch normalization layer"""
        reuse_flag = True if self.is_train is False else None
        return tf.contrib.layers.batch_norm(inputs=input_layer,
                                            decay=0.99,
                                            center=True,
                                            scale=True,
                                            is_training=self.is_train,
                                            reuse=reuse_flag)

    def add_conv_layer(self, input_layer, hyperparams, func='relu', bn=True):
        """Convolution Layer with hyperparamters and activation and batch
        normalization after nonlinearity as opposed to before nonlinearity as
        cited in Ioffe and Szegedy 2015."""
        W = self._weight_variable(shape=hyperparams[0])
        b = self._bias_variable(shape=hyperparams[1])
        if bn:
            return self._batch_normalize(
                   self._nonlinearity(func)(self._conv2d(input_layer, W) + b))
        elif not bn:
            return self._nonlinearity(func)(self._conv2d(input_layer, W) + b)

    def add_pooling_layer(self, input_layer):
        """max pooling layer to reduce overfitting"""
        return self._max_pool(input_layer)

    def add_dense_layer(self, input_layer, hyperparams, func='relu', bn=True):
        """Densely Connected Layer with hyperparamters and activation. Batch
        normalization inserted after nonlinearity as opposed to before as
        cited in Ioffe and Szegedy 2015."""
        W = self._weight_variable(shape=hyperparams[0])
        b = self._bias_variable(shape=hyperparams[1])
        x_ravel = tf.reshape(input_layer, shape=[-1, hyperparams[0][0]])
        if bn:
            return self._batch_normalize(
                   self._nonlinearity(func)(tf.matmul(x_ravel, W) + b))
        elif not bn:
            return self._nonlinearity(func)(tf.matmul(x_ravel, W) + b)

    def add_drop_out_layer(self, input_layer):
        """drop out layer to reduce overfitting"""
        return tf.nn.dropout(input_layer, self.keep_prob)

    def add_read_out_layer(self, input_layer):
        """read out layer with output shape of [batch_size, num_classes]
        in order to feed into softmax"""
        input_layer_m = int(input_layer.get_shape()[1])
        W = self._weight_variable(shape=[input_layer_m, self._n_class])
        b = self._bias_variable(shape=[self._n_class])

        return tf.matmul(input_layer, W) + b
