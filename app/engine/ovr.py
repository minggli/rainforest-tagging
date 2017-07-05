# -*- coding: utf-8 -*-
"""
ovr

One-versus-Rest approach for multi-label classification, predicting each label
independently using Convolutional Neural Network similar architecture to VGG-32

VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
Simonyan K. & Zisserman A. (2015)

Densely Connected Convolutional Networks
Huang et al 2016
"""

import tensorflow as tf

import numpy as np

from __main__ import EVAL, TRAIN, ENSEMBLE
from app.models.cnn import BasicCNN, DenseNet
from app.settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                          MAX_STEPS, ALPHA, BETA, TAGS, TAGS_WEIGHTINGS, EXT,
                          TAGS_THRESHOLDS, VALID_SIZE, KEEP_RATE, OUTPUT_PATH,
                          N_THREADS, AUGMENT)
from app.pipeline import data_pipe, generate_data_skeleton
from app.controllers import (train, save_session, predict, restore_session,
                             submit)


# def vgg_16_train(class_balance, l2_norm):
#
#     global prediction, loss, train_step, accuracy, saver, is_train
#
#     conv_1 = \
#         cnn.add_conv_layer(image_feed, [[3, 3, IMAGE_SHAPE[-1], 32], [32]])
#     conv_2 = cnn.add_conv_layer(conv_1, [[3, 3, 32, 32], [32]])
#     max_pool_1 = cnn.add_pooling_layer(conv_2)
#     conv_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 32, 64], [64]])
#     conv_4 = cnn.add_conv_layer(conv_3, [[3, 3, 64, 64], [64]])
#     max_pool_2 = cnn.add_pooling_layer(conv_4)
#     conv_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 64, 128], [128]])
#     conv_6 = cnn.add_conv_layer(conv_5, [[3, 3, 128, 128], [128]])
#     conv_7 = cnn.add_conv_layer(conv_6, [[3, 3, 128, 128], [128]])
#     max_pool_3 = cnn.add_pooling_layer(conv_7)
#     conv_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 128, 256], [256]])
#     conv_9 = cnn.add_conv_layer(conv_8, [[3, 3, 256, 256], [256]])
#     conv_10 = cnn.add_conv_layer(conv_9, [[3, 3, 256, 256], [256]])
#     max_pool_4 = cnn.add_pooling_layer(conv_10)
#     conv_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 256, 256], [256]])
#     conv_12 = cnn.add_conv_layer(conv_11, [[3, 3, 256, 256], [256]])
#     conv_13 = cnn.add_conv_layer(conv_12, [[3, 3, 256, 256], [256]])
#     max_pool_5 = cnn.add_pooling_layer(conv_13)
#     dense_1 = cnn.add_dense_layer(max_pool_5, [[4 * 4 * 256, 2048], [2048]])
#     drop_out_1 = cnn.add_drop_out_layer(dense_1)
#     dense_2 = cnn.add_dense_layer(drop_out_1, [[2048, 512], [512]])
#     drop_out_2 = cnn.add_drop_out_layer(dense_2)
#     logits = cnn.add_read_out_layer(drop_out_2)
#
#     cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
#                                                             labels=label_feed)
#
#     if class_balance:
#         class_weights = tf.constant([[TAGS_WEIGHTINGS]],
#                                     shape=[1, cnn._n_class])
#         cross_entropy *= class_weights
#
#     if l2_norm:
#         weights2norm = [var for var in tf.trainable_variables()
#                         if var.name.startswith(('weight', 'bias'))][-32:]
#         regularizers = tf.add_n([tf.nn.l2_loss(var) for var in weights2norm])
#         cross_entropy += BETA * regularizers
#
#     for n in tf.global_variables():
#         print(n)
#
#     loss = tf.reduce_mean(cross_entropy)
#
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     with tf.control_dependencies(update_ops):
#         train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA,
#                                                decay=0.7,
#                                                momentum=.5,
#                                                epsilon=1e-10,
#                                                use_locking=False,
#                                                centered=False).minimize(loss)
#     prediction = tf.nn.sigmoid(logits)
#     correct_pred = tf.equal(tf.cast(prediction > TAGS_THRESHOLDS, tf.int8),
#                             tf.cast(label_feed, tf.int8))
#     all_correct_pred = tf.reduce_min(tf.cast(correct_pred, tf.float32), 1)
#     accuracy = tf.reduce_mean(all_correct_pred)
#
#     saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())

#
# def vgg_16_eval():
#
#     global prediction, saver
#
#     conv_1 = \
#         cnn.add_conv_layer(image_feed, [[3, 3, IMAGE_SHAPE[-1], 32], [32]])
#     conv_2 = cnn.add_conv_layer(conv_1, [[3, 3, 32, 32], [32]])
#     max_pool_1 = cnn.add_pooling_layer(conv_2)
#     conv_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 32, 64], [64]])
#     conv_4 = cnn.add_conv_layer(conv_3, [[3, 3, 64, 64], [64]])
#     max_pool_2 = cnn.add_pooling_layer(conv_4)
#     conv_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 64, 128], [128]])
#     conv_6 = cnn.add_conv_layer(conv_5, [[3, 3, 128, 128], [128]])
#     conv_7 = cnn.add_conv_layer(conv_6, [[3, 3, 128, 128], [128]])
#     max_pool_3 = cnn.add_pooling_layer(conv_7)
#     conv_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 128, 256], [256]])
#     conv_9 = cnn.add_conv_layer(conv_8, [[3, 3, 256, 256], [256]])
#     conv_10 = cnn.add_conv_layer(conv_9, [[3, 3, 256, 256], [256]])
#     max_pool_4 = cnn.add_pooling_layer(conv_10)
#     conv_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 256, 256], [256]])
#     conv_12 = cnn.add_conv_layer(conv_11, [[3, 3, 256, 256], [256]])
#     conv_13 = cnn.add_conv_layer(conv_12, [[3, 3, 256, 256], [256]])
#     max_pool_5 = cnn.add_pooling_layer(conv_13)
#     dense_1 = cnn.add_dense_layer(max_pool_5, [[4 * 4 * 256, 2048], [2048]])
#     drop_out_1 = cnn.add_drop_out_layer(dense_1)
#     dense_2 = cnn.add_dense_layer(drop_out_1, [[2048, 512], [512]])
#     drop_out_2 = cnn.add_drop_out_layer(dense_2)
#     logits = cnn.add_read_out_layer(drop_out_2)
#
#     prediction = tf.nn.sigmoid(logits)
#
#     # without saver object restore doesn't actually work.
#     saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())


def densenet(class_balance=False, l2_norm=False):
    """DenseNet-BC 121"""
    global prediction, loss, train_step, accuracy, saver, is_train

    init_conv = dn.add_conv_layer(
                            image_feed,
                            [[7, 7, IMAGE_SHAPE[-1], 2 * dn._k], [2 * dn._k]],
                            bn=False)
    init_pool = dn.add_pooling_layer(init_conv, kernel_size=[1, 3, 3, 1])
    dense_block_1 = dn.add_dense_block(init_pool, L=3)
    transition_layer_1 = dn.add_transition_layer(dense_block_1)
    dense_block_2 = dn.add_dense_block(transition_layer_1, L=6)
    transition_layer_2 = dn.add_transition_layer(dense_block_2)
    dense_block_3 = dn.add_dense_block(transition_layer_2, L=18)
    transition_layer_3 = dn.add_transition_layer(dense_block_3)
    dense_block_4 = dn.add_dense_block(transition_layer_3, L=12)
    global_pool = dn.add_global_average_pool(dense_block_4)
    dim = int(global_pool.get_shape()[-1])
    dense_layer_1 = dn.add_dense_layer(global_pool, [[dim, 1000], [1000]],
                                       bn=False)
    drop_out_1 = dn.add_drop_out_layer(dense_layer_1)
    logits = dn.add_read_out_layer(drop_out_1)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=label_feed)
    # Explicit logistic log-loss function
    # cross_entropy = - (y_ * tf.log(1 / (1 + tf.exp(-logits)) + 1e-9) +
    #                  (1 - y_) * tf.log(1 - 1 / (1 + tf.exp(-logits)) + 1e-9))

    if class_balance:
        class_weights = tf.constant([[TAGS_WEIGHTINGS]],
                                    shape=[1, dn._n_class])
        cross_entropy *= class_weights

    if l2_norm:
        weights2norm = [var for var in tf.trainable_variables()
                        if var.name.startswith(('weight', 'bias'))][-6:]
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in weights2norm])
        cross_entropy += BETA * regularizers

    for n in tf.global_variables():
        print(n)

    loss = tf.reduce_mean(cross_entropy)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA,
                                               decay=0.7,
                                               momentum=.5,
                                               epsilon=1e-10,
                                               use_locking=False,
                                               centered=False).minimize(loss)
    prediction = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.cast(prediction > TAGS_THRESHOLDS, tf.int8),
                            tf.cast(label_feed, tf.int8))
    all_correct_pred = tf.reduce_min(tf.cast(correct_pred, tf.float32), 1)
    accuracy = tf.reduce_mean(all_correct_pred)

    saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())


def densenet_eval():
    """DenseNet-BC 121"""
    global prediction, saver

    init_conv = dn.add_conv_layer(
                            image_feed,
                            [[7, 7, IMAGE_SHAPE[-1], 2 * dn._k], [2 * dn._k]],
                            bn=False)
    init_pool = dn.add_pooling_layer(init_conv, kernel_size=[1, 3, 3, 1])
    dense_block_1 = dn.add_dense_block(init_pool, L=3)
    transition_layer_1 = dn.add_transition_layer(dense_block_1)
    dense_block_2 = dn.add_dense_block(transition_layer_1, L=6)
    transition_layer_2 = dn.add_transition_layer(dense_block_2)
    dense_block_3 = dn.add_dense_block(transition_layer_2, L=18)
    transition_layer_3 = dn.add_transition_layer(dense_block_3)
    dense_block_4 = dn.add_dense_block(transition_layer_3, L=12)
    global_pool = dn.add_global_average_pool(dense_block_4)
    dim = int(global_pool.get_shape()[-1])
    dense_layer_1 = dn.add_dense_layer(global_pool, [[dim, 1000], [1000]],
                                       bn=False)
    drop_out_1 = dn.add_drop_out_layer(dense_layer_1)
    logits = dn.add_read_out_layer(drop_out_1)

    prediction = tf.nn.sigmoid(logits)
    saver = tf.train.Saver(max_to_keep=5, var_list=tf.global_variables())


train_file_array, train_label_array, valid_file_array, valid_label_array = \
                    generate_data_skeleton(root_dir=IMAGE_PATH + 'train',
                                           valid_size=VALID_SIZE, ext=EXT)
test_file_array, dummy_label_array = \
                    generate_data_skeleton(root_dir=IMAGE_PATH + 'test',
                                           valid_size=None, ext=EXT)

ensemble_probs = list()

for iteration in range(ENSEMBLE):
    if TRAIN:
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) \
                as sess, tf.device('/cpu:0'):
            train_image_batch, train_label_batch = data_pipe(
                                                        train_file_array,
                                                        train_label_array,
                                                        num_epochs=None,
                                                        shape=IMAGE_SHAPE,
                                                        batch_size=BATCH_SIZE,
                                                        # no aug given bn
                                                        augmentation=AUGMENT,
                                                        shuffle=True,
                                                        threads=N_THREADS)
            valid_image_batch, valid_label_batch = data_pipe(
                                                        valid_file_array,
                                                        valid_label_array,
                                                        num_epochs=None,
                                                        shape=IMAGE_SHAPE,
                                                        batch_size=BATCH_SIZE,
                                                        augmentation=AUGMENT,
                                                        shuffle=True,
                                                        threads=N_THREADS)

            # cnn = BasicCNN(IMAGE_SHAPE, 17, keep_prob=KEEP_RATE)
            # is_train = cnn.is_train

            dn = DenseNet(IMAGE_SHAPE,
                          num_classes=17,
                          keep_prob=KEEP_RATE,
                          growth=32,
                          bottleneck=4,
                          compression=.5)
            is_train = dn.is_train

            # !!! inefficient feeding of data despite 90%+ GPU utilisation
            image_feed = tf.cond(is_train,
                                 lambda: train_image_batch,
                                 lambda: valid_image_batch)
            label_feed = tf.cond(is_train,
                                 lambda: train_label_batch,
                                 lambda: valid_label_batch)
            # image_feed = train_image_batch
            # label_feed = train_label_batch

            with tf.device('/gpu:0'):
                densenet(class_balance=False, l2_norm=False)
                # vgg_16_train(class_balance=False, l2_norm=False)

            init_op = tf.group(tf.local_variables_initializer(),
                               tf.global_variables_initializer())

            sess.run(init_op)
            # sess.graph.finalize()
            train(MAX_STEPS, sess, is_train, prediction, label_feed,
                  train_step, accuracy, loss, TAGS_THRESHOLDS)
            save_session(sess, path=MODEL_PATH, sav=saver)

    if EVAL:
        tf.reset_default_graph()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) \
                as sess, tf.device('/cpu:0'):
            test_image_batch, _ = data_pipe(
                                                        test_file_array,
                                                        dummy_label_array,
                                                        num_epochs=1,
                                                        shape=IMAGE_SHAPE,
                                                        batch_size=BATCH_SIZE,
                                                        augmentation=AUGMENT,
                                                        shuffle=False)

            # cnn = BasicCNN(IMAGE_SHAPE, 17, keep_prob=KEEP_RATE)
            image_feed = test_image_batch
            dn = DenseNet(IMAGE_SHAPE,
                          num_classes=17,
                          keep_prob=KEEP_RATE,
                          growth=32,
                          bottleneck=4,
                          compression=.5)
            with tf.device('/gpu:0'):
                # vgg_16_eval()
                densenet_eval()

            init_op = tf.group(tf.local_variables_initializer(),
                               tf.global_variables_initializer())

            sess.run(init_op)
            restore_session(sess, MODEL_PATH)
            probs = predict(sess, prediction)
            ensemble_probs.append(probs)

if EVAL:
    final_probs = np.mean(ensemble_probs, axis=0)
    submit(final_probs, OUTPUT_PATH, TAGS, TAGS_THRESHOLDS)
