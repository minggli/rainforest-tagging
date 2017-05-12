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

from ..main import EVAL, TRAIN
from ..models.cnn import ConvolutionalNeuralNetwork
# from ..models.rnn import LSTM
# from ..label2vec import LabelVectorizer
from ..settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                        MAX_STEPS, ALPHA, BETA, TAGS, TAGS_WEIGHTINGS)
from ..pipeline import data_pipe, generate_data_skeleton
from ..controllers import (train, save_session, predict, submit,
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



# # Tensorflow loss function API
# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
#                                                         labels=y_)
# # Explicit logistic loss function
# # cross_entropy = - (y_ * tf.log(1 / (1 + tf.exp(-logits)) + 1e-9) +
# #                    (1 - y_) * tf.log(1 - 1 / (1 + tf.exp(-logits)) + 1e-9))
# # [batch_size, 17] of logistic loss for each of 17 classes
#
# # applying label weights to loss function
# if True:
#     class_weights = tf.constant([[TAGS_WEIGHTINGS]], shape=[1, 17])
#     cross_entropy *= class_weights
#
# # add L2 regularization on weights from readout layer and dense layers
# if True:
#     weights2norm = [var for var in tf.trainable_variables()
#                     if var.name.startswith(('weight', 'bias'))][-6:]
#     regularizers = tf.add_n([tf.nn.l2_loss(var) for var in weights2norm])
#     cross_entropy += BETA * regularizers
#
# loss = tf.reduce_mean(cross_entropy)
#
# for var in tf.trainable_variables():
#     print(var)
#
# # Numerical Optimisation
# train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA,
#                                        decay=0.9,
#                                        momentum=.5,
#                                        epsilon=1e-10,
#                                        use_locking=False,
#                                        centered=False).minimize(loss)
#
# # eval
# correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(logits)), y_)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # saver
# saver = tf.train.Saver(max_to_keep=5, var_list=tf.trainable_variables())
#
# # session
# sess = tf.Session()
#
# if TRAIN:
#     # prepare data feed
#     train_file_array, train_label_array, valid_file_array, valid_label_array =\
#         generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.15)
#     train_image_batch, train_label_batch = data_pipe(
#                                             train_file_array,
#                                             train_label_array,
#                                             num_epochs=None,
#                                             shape=IMAGE_SHAPE,
#                                             batch_size=BATCH_SIZE,
#                                             shuffle=True)
#     valid_image_batch, valid_label_batch = data_pipe(
#                                             valid_file_array,
#                                             valid_label_array,
#                                             num_epochs=None,
#                                             shape=IMAGE_SHAPE,
#                                             batch_size=BATCH_SIZE,
#                                             shuffle=True)
#
#     init_op = tf.group(tf.local_variables_initializer(),
#                        tf.global_variables_initializer())
#     sess.run(init_op)
#
#     with sess:
#         train(MAX_STEPS, sess, x, y_, keep_prob, logits, train_image_batch,
#               train_label_batch, valid_image_batch, valid_label_batch,
#               train_step, accuracy, loss)
#         save_session(sess, path=MODEL_PATH, sav=saver)
#
# if EVAL:
#
#     test_file_array, _ = \
#         generate_data_skeleton(root_dir=IMAGE_PATH + 'test', valid_size=None)
#     # no shuffling or more than 1 epoch of test set, only through once.
#     test_image_batch = data_pipe(
#                             test_file_array,
#                             _,
#                             num_epochs=1,
#                             shape=IMAGE_SHAPE,
#                             batch_size=BATCH_SIZE,
#                             shuffle=False)[0]
#
#     # only need to initiate data pipeline stored in local variable
#     sess.run(tf.local_variables_initializer())
#
#     with sess:
#         restore_session(sess, MODEL_PATH)
#         probs = predict(sess, x, keep_prob, logits, test_image_batch, TAGS)
#         submit(probs, IMAGE_PATH + 'test')
#
# # delete session manually to prevent exit error.
# del sess
