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

# from __main__ import EVAL, TRAIN, ENSEMBLE
from app.models.cnn import BasicCNN, DenseNet
from app.settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                          MAX_STEPS, ALPHA, BETA, TAGS, TAGS_WEIGHTINGS, EXT,
                          TAGS_THRESHOLDS, VALID_SIZE, KEEP_RATE, OUTPUT_PATH,
                          N_THREADS, AUGMENT)
from app.pipeline import data_pipe, generate_data_skeleton
from app.controllers import (train, save_session, predict, restore_session,
                             submit)
#
# cnn = BasicCNN(IMAGE_SHAPE, 17, keep_prob=KEEP_RATE)
#
# conv_1 = cnn.add_conv_layer(image_feed, [[3, 3, IMAGE_SHAPE[-1], 32], [32]])
# conv_2 = cnn.add_conv_layer(conv_1, [[3, 3, 32, 32], [32]])
# max_pool_1 = cnn.add_pooling_layer(conv_2)
# conv_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 32, 64], [64]])
# conv_4 = cnn.add_conv_layer(conv_3, [[3, 3, 64, 64], [64]])
# max_pool_2 = cnn.add_pooling_layer(conv_4)
# conv_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 64, 128], [128]])
# conv_6 = cnn.add_conv_layer(conv_5, [[3, 3, 128, 128], [128]])
# conv_7 = cnn.add_conv_layer(conv_6, [[3, 3, 128, 128], [128]])
# max_pool_3 = cnn.add_pooling_layer(conv_7)
# conv_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 128, 256], [256]])
# conv_9 = cnn.add_conv_layer(conv_8, [[3, 3, 256, 256], [256]])
# conv_10 = cnn.add_conv_layer(conv_9, [[3, 3, 256, 256], [256]])
# max_pool_4 = cnn.add_pooling_layer(conv_10)
# conv_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 256, 256], [256]])
# conv_12 = cnn.add_conv_layer(conv_11, [[3, 3, 256, 256], [256]])
# conv_13 = cnn.add_conv_layer(conv_12, [[3, 3, 256, 256], [256]])
# max_pool_5 = cnn.add_pooling_layer(conv_13)
# dense_1 = cnn.add_dense_layer(max_pool_5, [[4 * 4 * 256, 2048], [2048]])
# drop_out_1 = cnn.add_drop_out_layer(dense_1)
# dense_2 = cnn.add_dense_layer(drop_out_1, [[2048, 512], [512]])
# drop_out_2 = cnn.add_drop_out_layer(dense_2)
# image_vector = cnn.add_read_out_layer(drop_out_2)
# [batch_size, 128] vector representation of image

# using Tensorflow API first before using self-implemented lstm module

# label embedding in (17, 300)
label_embedding = LabelVectorizer().fit(TAGS).transform()
Ul = tf.constant(label_embedding, name='label_embedding')
# word_vector = tf.nn.embedding_lookup(Ul, tf.where(tf.equal(label_feed, 1)))
word_vector = tf.matmul(label_feed, Ul)
# [batch_size * 17], [17, 300]
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
