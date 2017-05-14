# -*- coding: utf-8 -*-
"""
ovr

One-versus-Rest approach for multi-label classification, predicting each label
independently after CNN (VGG-16)

VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION
Simonyan K. & Zisserman A. (2015)
"""

import tensorflow as tf
import numpy as np
from ..main import EVAL, TRAIN, ENSEMBLE
from ..models.cnn import ConvolutionalNeuralNetwork
from ..settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                        MAX_STEPS, ALPHA, BETA, TAGS, TAGS_WEIGHTINGS,
                        TAGS_THRESHOLDS)
from ..pipeline import data_pipe, generate_data_skeleton
from ..controllers import train, save_session, predict, restore_session, submit


def vgg_16(class_balance, l2_norm):

    global x, y_, keep_prob, logits, loss, train_step, accuracy, saver

    cnn = ConvolutionalNeuralNetwork(shape=IMAGE_SHAPE, num_classes=17)
    x, y_, keep_prob = cnn.x, cnn.y_, cnn.keep_prob

    conv_layer_1 = cnn.add_conv_layer(x, [[3, 3, IMAGE_SHAPE[-1], 6], [6]])
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
    fc1 = cnn.add_dense_layer(max_pool_5, [[1 * 1 * 48, 2048], [2048]])
    # drop_out_layer_1 = cnn.add_drop_out_layer(fc1, keep_prob)
    fc2 = cnn.add_dense_layer(fc1, [[2048, 1024], [1024]])
    # drop_out_layer_2 = cnn.add_drop_out_layer(fc2, keep_prob)
    logits = cnn.add_read_out_layer(fc2)
    # [batch_size, 17] of logits (θ transpose X) for each of 17 classes

    # Tensorflow loss function API
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                            labels=y_)
    # Explicit logistic loss function
    # cross_entropy = - (y_ * tf.log(1 / (1 + tf.exp(-logits)) + 1e-9) * .75 +
    #                  (1 - y_) * tf.log(1 - 1 / (1 + tf.exp(-logits)) + 1e-9))
    # [batch_size, 17] of logistic loss for each of 17 classes

    # applying label weights to loss function
    if class_balance:
        class_weights = tf.constant([[TAGS_WEIGHTINGS]],
                                    shape=[1, cnn._n_class])
        cross_entropy *= class_weights

    # add L2 regularization on weights from readout layer and dense layers
    if l2_norm:
        weights2norm = [var for var in tf.trainable_variables()
                        if var.name.startswith(('weight', 'bias'))][-4:]
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in weights2norm])
        cross_entropy += BETA * regularizers

    loss = tf.reduce_mean(cross_entropy)

    # Numerical Optimisation
    train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA,
                                           decay=0.9,
                                           momentum=.5,
                                           epsilon=1e-10,
                                           use_locking=False,
                                           centered=False).minimize(loss)

    # eval
    correct_pred = tf.equal(tf.round(tf.nn.sigmoid(logits)), y_)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    all_correct_pred = tf.reduce_min(tf.cast(correct_pred, tf.float32), 1)
    accuracy = tf.reduce_mean(all_correct_pred)

    # saver
    saver = tf.train.Saver(max_to_keep=5, var_list=tf.trainable_variables())


ensemble_probs = list()

for iteration in range(ENSEMBLE):
    tf.reset_default_graph()
    vgg_16(class_balance=False, l2_norm=False)

    if TRAIN:
        with tf.Session() as sess:
            train_file_array, train_label_array, valid_file_array,\
                valid_label_array = generate_data_skeleton(
                                                root_dir=IMAGE_PATH + 'train',
                                                valid_size=.15,
                                                ext=('.png', '.csv'))
            train_image_batch, train_label_batch = data_pipe(
                                                train_file_array,
                                                train_label_array,
                                                num_epochs=None,
                                                shape=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
            valid_image_batch, valid_label_batch = data_pipe(
                                                valid_file_array,
                                                valid_label_array,
                                                num_epochs=None,
                                                shape=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)

            init_op = tf.group(tf.local_variables_initializer(),
                               tf.global_variables_initializer())
            sess.run(init_op)

            train(MAX_STEPS, sess, x, y_, keep_prob, logits, train_image_batch,
                  train_label_batch, valid_image_batch, valid_label_batch,
                  train_step, accuracy, loss, TAGS_THRESHOLDS)
            save_session(sess, path=MODEL_PATH, sav=saver)

    if EVAL:
        with tf.Session() as sess:
            test_file_array, _ = generate_data_skeleton(
                                                root_dir=IMAGE_PATH + 'test',
                                                valid_size=None,
                                                ext=('.png', '.csv'))
            # !!! no shuffling and only 1 epoch of test set.
            test_image_batch, _ = data_pipe(
                                                test_file_array,
                                                _,
                                                num_epochs=1,
                                                shape=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                shuffle=False)

            sess.run(tf.local_variables_initializer())

            restore_session(sess, MODEL_PATH)
            probs = predict(sess, x, keep_prob, logits, test_image_batch)
            print(probs.shape)
            ensemble_probs.append(probs)


if EVAL:
    final_probs = np.mean(ensemble_probs, axis=0)
    submit(final_probs, IMAGE_PATH + 'test', TAGS, TAGS_THRESHOLDS)

# delete session manually to prevent exit error.
del sess