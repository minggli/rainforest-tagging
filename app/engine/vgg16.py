# -*- coding: utf-8 -*-

import tensorflow as tf

from ..main import EVAL, TRAIN
from ..models.cnn import ConvolutionalNeuralNet
from ..settings import (IMAGE_PATH, IMAGE_SHAPE, BATCH_SIZE, MODEL_PATH,
                        MAX_STEPS, ALPHA, BETA)
from ..pipeline import data_pipe, generate_data_skeleton
from ..controllers import (train, save_session, predict, submit,
                           restore_session)

sess = tf.Session()
cnn = ConvolutionalNeuralNet(shape=IMAGE_SHAPE)

x, _y = cnn.x, cnn._y
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
fc1 = cnn.add_dense_layer(max_pool_5, [[3 * 5 * 48, 256], [256],
                                       [-1, 3 * 5 * 48]])
# drop_out_layer_1 = cnn.add_drop_out_layer(fc1, keep_prob)
fc2 = cnn.add_dense_layer(fc1, [[256, 128], [128], [-1, 256]])
# drop_out_layer_2 = cnn.add_drop_out_layer(fc2, keep_prob)
logits = cnn.add_read_out_layer(fc2, [[128, 8], [8]])
# [batch_size, 8]

# default loss function
cross_entropy = \
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=_y)
loss = tf.reduce_mean(cross_entropy)

# applying label weights to loss function
if False:
    class_weight = tf.constant([[0.545, 0.947, 0.969, 0.982, 0.877, 0.921,
                                 0.953, 0.806]])
    weight_per_label = tf.transpose(tf.matmul(_y, tf.transpose(class_weight)))
    loss = tf.reduce_mean(tf.multiply(weight_per_label, cross_entropy))

# add L2 regularization on weights from readout layer
if True:
    out_weights = [var for var in tf.trainable_variables()
                   if var.name.startswith('Variable_')][-2]
    regularizer = tf.nn.l2_loss(out_weights)
    loss = tf.reduce_mean(loss + BETA * regularizer)

# train Ops
train_step = tf.train.RMSPropOptimizer(learning_rate=ALPHA).minimize(loss)

# eval
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver
saver = tf.train.Saver(max_to_keep=5, var_list=tf.trainable_variables())

if TRAIN:
    # prepare data feed
    train_file_array, train_label_array, valid_file_array, valid_label_array =\
        generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.15)
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

    with sess:
        train(MAX_STEPS, sess, x, _y, keep_prob, train_image_batch,
              train_label_batch, valid_image_batch, valid_label_batch,
              train_step, accuracy, loss)
        save_session(sess, path=MODEL_PATH, sav=saver)

if EVAL:

    test_file_array, _ = \
        generate_data_skeleton(root_dir=IMAGE_PATH + 'test_stg1',
                               valid_size=None)
    # no shuffling or more than 1 epoch of test set, only through once.
    test_image_batch = data_pipe(
                            test_file_array,
                            _,
                            num_epochs=1,
                            shape=IMAGE_SHAPE,
                            batch_size=BATCH_SIZE,
                            shuffle=False)[0]

    # only need to initiate data pipeline stored in local variable
    sess.run(tf.local_variables_initializer())

    with sess:
        restore_session(sess, MODEL_PATH)
        probs = predict(sess, x, keep_prob, logits, test_image_batch)
        submit(probs, IMAGE_PATH)

# delete session manually to prevent exit error.
del sess
