# -*- coding: utf-8 -*-
"""
cnn

abstraction built on top of tensorflow for constructing Convolutional Neural
Network
"""

import warnings

import tensorflow as tf


class _BaseCNN(object):

    def __init__(self, shape, num_classes, keep_prob):
        """shape: [n_samples, channels, n_features]"""
        self._shape = shape
        self._n_class = num_classes
        self._keep_rate = keep_prob
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
    def _max_pool(x, kernel_size):
        """max pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.max_pool(value=x,
                              ksize=kernel_size,
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def _average_pool(x, kernel_size):
        """avg pooling with kernal size 2x2 and slide by 2 pixels each time"""
        return tf.nn.avg_pool(value=x,
                              ksize=kernel_size,
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    @staticmethod
    def _nonlinearity(activation='relu'):
        if activation == 'sigmoid':
            return tf.nn.sigmoid
        elif activation == 'relu':
            return tf.nn.relu
        elif activation == 'tanh':
            return tf.tanh

    @property
    def x(self):
        """feature set"""
        warnings.warn("using placeholder to feed data will causing low "
                      "efficiency between Python and C++ interface",
                      RuntimeWarning)
        return tf.placeholder(dtype=tf.float32,
                              # transform 3D shape to 4D to include batch size
                              shape=(None, ) + self._shape,
                              name='feature')

    @property
    def y_(self):
        """ground truth, in one-hot format"""
        warnings.warn("using placeholder to feed data will causing low "
                      "efficiency between Python and C++ interface",
                      RuntimeWarning)
        return tf.placeholder(dtype=tf.float32,
                              shape=(None, self._n_class),
                              name='label')

    @property
    def _keep_prob(self):
        """the probability constant to keep output from previous layer."""
        return tf.cond(self.is_train, lambda: tf.constant(self._keep_rate),
                       lambda: tf.constant(1.))

    @property
    def _is_train(self):
        """indicates if network is under training mode, default False."""
        return tf.placeholder_with_default(input=False,
                                           shape=[],
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


class BasicCNN(_BaseCNN):
    def __init__(self, shape, num_classes, keep_prob=.5):
        super(BasicCNN, self).__init__(shape, num_classes, keep_prob)

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

    def add_pooling_layer(self,
                          input_layer,
                          kernel_size=[1, 2, 2, 1],
                          mode='max'):
        """max pooling layer to reduce overfitting"""
        if mode == 'max':
            return self._max_pool(input_layer, kernel_size)
        elif mode == 'average':
            return self._average_pool(input_layer, kernel_size)

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


class DenseNet(BasicCNN):
    """Implementation of Densely Connected Convolutional Networks by
    Huang et al 2016"""

    def __init__(self,
                 shape,
                 num_classes,
                 keep_prob=.5,
                 growth=12,
                 bottleneck=4,
                 compression=.5):
        """
        Growth rate (denoted as k) refers to added channels from each
        Dense Block where the paper suggests k=12 as default and indicates that
        k=40 outperforms previous state-of-the-art on CIFAR;

        Bottleneck (denoted as b) refers to additional 1x1 composite function
        within each Dense Block to cap number of input channels by b * k

        Compression (denoted as theta) improves 'model compactness' with (0, 1]
        compression factor, applied at transition layer to reduce channels
        """
        super(DenseNet, self).__init__(shape, num_classes, keep_prob)
        self._k = growth
        self._b = bottleneck
        self._theta = compression

    def _composite_function(self, input_layer, hyperparams, func='relu'):
        """Standard Composite Function H, consisting of Batch Normalization,
        Nonlinearity (e.g. RELU), Convolution with filter size 3x3 or 1x1."""
        W = self._weight_variable(shape=hyperparams[0])
        b = self._bias_variable(shape=hyperparams[1])

        _internal_state = self._batch_normalize(input_layer)
        _internal_state = self._nonlinearity(func)(_internal_state)
        _internal_state = self._conv2d(_internal_state, W)
        return _internal_state + b

    @staticmethod
    def concat(tuple_of_tensors):
        return tf.concat(values=tuple_of_tensors, axis=3)

    def add_dense_block(self, _internal_state, L):
        """Core element of DensetNet. Dense Block consists of densely connected
        composite functions with each taking in all previous outputs instead of
        just the previous layer within the Dense Block.

        Bottleneck and Compression (i.e.) DenseNet-BC
        structure yields superior results against benchmarks

        L denotes the total number of composite functions and must be >= 1."""

        p_bottleneck = [[1, 1, None, self._b * self._k], [self._b * self._k]]
        p_conv = [[3, 3, self._b * self._k, self._k], [self._k]]
        for l in range(L):
            # update bottleneck layer param to reflect growth
            p_bottleneck[0][2] = int(_internal_state.get_shape()[-1])
            _bottleneck_state = \
                self._composite_function(_internal_state, p_bottleneck)
            _new_state = \
                self._composite_function(_bottleneck_state, p_conv)
            # !!! perform concatenation along 4th dimension
            _internal_state = self.concat([_internal_state, _new_state])
        # final _internal_state in shape [BATCH_SIZE, height, width, +=k]
        return _internal_state

    def add_transition_layer(self, concatenated_input):
        """transition layer in-between dense block, with compression factor C
        builtin by default so DenseNet-C is implemented"""
        input_chl = int(concatenated_input.get_shape()[-1])
        output_chl = int(input_chl * self._theta)
        W = self._weight_variable(shape=[1, 1, input_chl, output_chl])
        b = self._bias_variable(shape=[output_chl])
        _internal_state = self._batch_normalize(concatenated_input)
        _internal_state = self._conv2d(_internal_state, W) + b
        _internal_state = self._average_pool(_internal_state, [1, 2, 2, 1])
        return _internal_state

    def add_global_average_pool(self, input_layer):
        h = int(input_layer.get_shape()[1])
        w = int(input_layer.get_shape()[2])
        kernel_size = [1, h, w, 1]
        return tf.nn.avg_pool(value=input_layer,
                              ksize=kernel_size,
                              strides=kernel_size,
                              padding='SAME')
