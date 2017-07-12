# -*- coding: utf-8 -*-
"""
jointnn (needs parameterising)

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
from app.label2vec import LabelVectorizer
from app.pipeline import data_pipe, generate_data_skeleton
from app.controllers import (train, save_session, predict, restore_session,
                             submit)

cnn = BasicCNN(IMAGE_SHAPE, 17, keep_prob=KEEP_RATE)

conv_1 = cnn.add_conv_layer(image_feed, [[3, 3, IMAGE_SHAPE[-1], 32], [32]])
conv_2 = cnn.add_conv_layer(conv_1, [[3, 3, 32, 32], [32]])
max_pool_1 = cnn.add_pooling_layer(conv_2)
conv_3 = cnn.add_conv_layer(max_pool_1, [[3, 3, 32, 64], [64]])
conv_4 = cnn.add_conv_layer(conv_3, [[3, 3, 64, 64], [64]])
max_pool_2 = cnn.add_pooling_layer(conv_4)
conv_5 = cnn.add_conv_layer(max_pool_2, [[3, 3, 64, 128], [128]])
conv_6 = cnn.add_conv_layer(conv_5, [[3, 3, 128, 128], [128]])
conv_7 = cnn.add_conv_layer(conv_6, [[3, 3, 128, 128], [128]])
max_pool_3 = cnn.add_pooling_layer(conv_7)
conv_8 = cnn.add_conv_layer(max_pool_3, [[3, 3, 128, 256], [256]])
conv_9 = cnn.add_conv_layer(conv_8, [[3, 3, 256, 256], [256]])
conv_10 = cnn.add_conv_layer(conv_9, [[3, 3, 256, 256], [256]])
max_pool_4 = cnn.add_pooling_layer(conv_10)
conv_11 = cnn.add_conv_layer(max_pool_4, [[3, 3, 256, 256], [256]])
conv_12 = cnn.add_conv_layer(conv_11, [[3, 3, 256, 256], [256]])
conv_13 = cnn.add_conv_layer(conv_12, [[3, 3, 256, 256], [256]])
max_pool_5 = cnn.add_pooling_layer(conv_13)
dense_1 = cnn.add_dense_layer(max_pool_5, [[4 * 4 * 256, 2048], [2048]])
drop_out_1 = cnn.add_drop_out_layer(dense_1)
dense_2 = cnn.add_dense_layer(drop_out_1, [[2048, 300], [300]])
drop_out_2 = cnn.add_drop_out_layer(dense_2)
image_vector = cnn.add_read_out_layer(drop_out_2)
# [batch_size, 300] vector representation of image

# !!! TODO need to re-order labels by frequency from default alphabetic order
label_embedding = LabelVectorizer().fit(TAGS).transform()
Ul = tf.constant(label_embedding, name='label_embedding_matrix')
# Ul label embedding in (17, 300)

word_vector = tf.nn.embedding_lookup(Ul, tf.where(tf.equal(label_feed, 1)))
# word_vector in shape [batch_size, num_labels, 300]
init = tf.truncated_normal_initializer(stddev=.1)
Uo = tf.get_variable(name='projection_matrix_rnn',
                     shape=[None, 300],
                     initializer=init)
Ui = tf.get_variable(name='projection_matrix_cnn',
                     shape=[None, 300],
                     initializer=init)

# require sequence to sequence configuration of LSTM to take previosuly
# predicted label (t - 1) as input for t
with tf.device('/gpu:0'):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(STATE_SIZE)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell,
                                              output_keep_prob=KEEP_RATE)
    # sent_length = size(word_vectors)
    outputs, final_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                                             encoder_inputs=,
                                             decoder_inputs=,
                                             cell=cell,
                                            num_encoder_symbols=enc_vocab_size+1,
                                            num_decoder_symbols=dec_vocab_size+2,
                                            embedding_size=enc_emb_size,
                                            feed_previous=True)
    # last = find_last(outputs, sent_length)
    # calculate simiarity per class between embedding matrix and prediction
    simiarity = Ul * last

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=label_feed)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    probs = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(probs, 1), tf.argmax(label_feed, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
