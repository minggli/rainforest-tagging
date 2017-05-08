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
from ..label2vec import LabelVectorizer
from ..settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                        MAX_STEPS, ALPHA, BETA, TAGS, TAGS_WEIGHTINGS)
from ..pipeline import data_pipe, generate_data_skeleton
from ..controllers import (train, save_session, predict, submit,
                           restore_session)

# Convolutional Neural Network as VGG-16
cnn = ConvolutionalNeuralNetwork(shape=IMAGE_SHAPE, num_classes=1)

with tf.variable_scope('VGG-16'):
    x, y_ = cnn.x, cnn.y_
    keep_prob = tf.placeholder(tf.float32)

    conv_layer_1 = cnn.add_conv_layer(x, [[3, 3, 3, 6], [6]])
    conv_layer_2 = cnn.add_conv_layer(conv_layer_1, [[3, 3, 6, 6], [6]])
    max_pool_1 = cnn.add_pooling_layer(conv_layer_2)
    conv_layer_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 6, 12], [12]])
    conv_layer_4 = cnn.add_conv_layer(conv_layer_3, [[3, 3, 12, 12], [12]])
    max_pool_2 = cnn.add_pooling_layer(conv_layer_4)
    conv_layer_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 12, 24], [24]])
    conv_layer_6 = cnn.add_conv_layer(conv_layer_5, [[3, 3, 24, 24], [24]])
    conv_layer_7 = cnn.add_conv_layer(conv_layer_6, [[3, 3, 24, 24], [24]])
    max_pool_3 = cnn.add_pooling_layer(conv_layer_7)
    conv_layer_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 24, 48], [48]])
    conv_layer_9 = cnn.add_conv_layer(conv_layer_8, [[3, 3, 48, 48], [48]])
    conv_layer_10 = cnn.add_conv_layer(conv_layer_9, [[3, 3, 48, 48], [48]])
    max_pool_4 = cnn.add_pooling_layer(conv_layer_10)
    conv_layer_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 48, 48], [48]])
    conv_layer_12 = cnn.add_conv_layer(conv_layer_11, [[3, 3, 48, 48], [48]])
    conv_layer_13 = cnn.add_conv_layer(conv_layer_12, [[3, 3, 48, 48], [48]])
    max_pool_5 = cnn.add_pooling_layer(conv_layer_13)
    fc1 = cnn.add_dense_layer(max_pool_5, [[4 * 4 * 48, 256], [256],
                                           [-1, 4 * 4 * 48]])
    img_vector = cnn.add_dense_layer(fc1, [[256, 128], [128], [-1, 256]])
    # [batch_size, 128] so image has vector representation of 128 dimensions.
    print(img_vector)

with tf.variable_scope('LSTM'):
    # label embedding in (17, 300)
    label_embedding = LabelVectorizer().fit(TAGS).transform()
    Ul = tf.constant(label_embedding, name='label_embedding')
    word_vector = tf.nn.embedding_lookup(Ul, tf.where(tf.equal(y_, 1)))
    # [batch_size, num_labels, 300]

    # using Tensorflow API first before using implemented rnn module
    weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=128,
                                        use_peepholes=False,
                                        initializer=weight_initializer,
                                        forget_bias=1.0,
                                        activation=tf.tanh)
    output, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=img_vector,
                                            sequence_length=None,
                                            initial_state=lstm_cell.zero_state)
    # [batch_size, 128] so each image has its label(s) represented as 128-d vector.
    #
    # w = U_label[y_.nonzero()]



# # default loss function
# cross_entropy = \
#         tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_)
# loss = tf.reduce_mean(cross_entropy)
#
# # applying label weights to loss function
# if True:
#     class_weight = tf.constant([TAGS_WEIGHTINGS])
#     # weight_per_label = tf.transpose(tf.multiply(y_, class_weight))
#     loss = tf.reduce_mean(class_weight * cross_entropy)
#
# # add L2 regularization on weights from readout layer
# if False:
#     out_weights = [var for var in tf.trainable_variables()
#                    if var.name.startswith('Variable_')][-2]
#     regularizer = tf.nn.l2_loss(out_weights)
#     loss = tf.reduce_mean(loss + BETA * regularizer)
#
# # train Ops
# train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA).minimize(loss)
#
# # eval
# correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(logits)), tf.round(y_))
# # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy = tf.reduce_mean(
#                   tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1))
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
#         train(MAX_STEPS, sess, x, y_, keep_prob, train_image_batch,
#               train_label_batch, valid_image_batch, valid_label_batch,
#               train_step, accuracy, loss)
#         save_session(sess, path=MODEL_PATH, sav=saver)
#
# if EVAL:
#
#     test_file_array, _ = \
#         generate_data_skeleton(root_dir=IMAGE_PATH + 'test',
#                                valid_size=None)
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
