# -*- coding: utf-8 -*-
"""
jointnn

joint embedding with Convolutional Neural Network and LSTM as inspired from:

CNN-RNN: A Unified Framework for Multi-label Image Classification
Wang et al (2015)

Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models
Kiros et al (2014)
"""

import tensorflow as tf

from app.main import EVAL, TRAIN
from app.models.cnn import ConvolutionalNeuralNetwork
# from app.models.rnn import LSTM
# from app.label2vec import LabelVectorizer
from app.settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                        MAX_STEPS, ALPHA, BETA, TAGS, TAGS_WEIGHTINGS)
from app.pipeline import data_pipe, generate_data_skeleton
from app.controllers import (train, save_session, predict, submit,
                           restore_session)

# Convolutional Neural Network as VGG-16
with tf.variable_scope('VGG-16'):
    cnn = ConvolutionalNeuralNetwork(shape=IMAGE_SHAPE, num_classes=None)
    x, keep_prob = cnn.x, cnn.keep_prob

    conv_layer_1 = cnn.add_conv_layer(x, [[3, 3, IMAGE_SHAPE[-1], 18], [18]])
    conv_layer_2 = cnn.add_conv_layer(conv_layer_1, [[3, 3, 18, 18], [18]])
    max_pool_1 = cnn.add_pooling_layer(conv_layer_2)
    conv_layer_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 18, 24], [24]])
    conv_layer_4 = cnn.add_conv_layer(conv_layer_3, [[3, 3, 24, 24], [24]])
    max_pool_2 = cnn.add_pooling_layer(conv_layer_4)
    conv_layer_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 24, 36], [36]])
    conv_layer_6 = cnn.add_conv_layer(conv_layer_5, [[3, 3, 36, 36], [36]])
    conv_layer_7 = cnn.add_conv_layer(conv_layer_6, [[3, 3, 36, 36], [36]])
    max_pool_3 = cnn.add_pooling_layer(conv_layer_7)
    conv_layer_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 36, 48], [48]])
    conv_layer_9 = cnn.add_conv_layer(conv_layer_8, [[3, 3, 48, 48], [48]])
    conv_layer_10 = cnn.add_conv_layer(conv_layer_9, [[3, 3, 48, 48], [48]])
    max_pool_4 = cnn.add_pooling_layer(conv_layer_10)
    conv_layer_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 48, 48], [48]])
    conv_layer_12 = cnn.add_conv_layer(conv_layer_11, [[3, 3, 48, 48], [48]])
    conv_layer_13 = cnn.add_conv_layer(conv_layer_12, [[3, 3, 48, 48], [48]])
    max_pool_5 = cnn.add_pooling_layer(conv_layer_13)
    fc1 = cnn.add_dense_layer(max_pool_5, [[1 * 1 * 48, 256], [256]])
    img_vector = cnn.add_dense_layer(fc1, [[256, 128], [128]])
    # [batch_size, 128] vector representation of image


with tf.variable_scope('LSTM'):
    # using Tensorflow API first before using self-implemented lstm module

    # label embedding in (17, 300)
    # label_embedding = LabelVectorizer().fit(TAGS).transform()
    # Ul = tf.constant(label_embedding, name='label_embedding')
    # word_vector = tf.nn.embedding_lookup(Ul, tf.where(tf.equal(y_, 1)))
    # [batch_size, num_labels from 1 to 17, 300]

    weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=300,
                                        use_peepholes=False,
                                        initializer=weight_initializer,
                                        forget_bias=1.0,
                                        activation=tf.tanh)
    output, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=img_vector,
                                            sequence_length=None,
                                            initial_state=lstm_cell.zero_state)
