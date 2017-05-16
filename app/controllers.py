# -*- coding: utf-8 -*-
"""
controllers

handle tensorflow session relating to ConvNet
"""
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import metrics
from datetime import datetime

from .pipeline import multithreading


def timeit(func):
    """calculate time for a function to complete"""

    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print('function {0} took {1:0.3f} s'.format(
              func.__name__, (end - start) * 1))
        return output
    return wrapper


@timeit
@multithreading
def train(n, sess, x, y_, keep_prob, is_train, logits, train_image_batch,
          train_label_batch, valid_image_batch, valid_label_batch, optimiser,
          metric, loss, thresholds):
    """train neural network and produce accuracies with validation set."""

    for global_step in range(n):
        train_image, train_label = \
                            sess.run([train_image_batch, train_label_batch])
        _, train_accuracy, train_loss, y_pred = sess.run(
                fetches=[optimiser, metric, loss, tf.nn.sigmoid(logits)],
                feed_dict={x: train_image, y_: train_label, keep_prob: .75,
                           is_train: True})
        f2_score = metrics.fbeta_score(y_true=train_label,
                                       y_pred=y_pred > thresholds,
                                       beta=2,
                                       average='samples')
        print("step {0} of {3}, train accuracy: {1:.4f}, F2 score: {4:.4f}"
              " log loss: {2:.4f}".format(global_step, train_accuracy,
                                          train_loss, n, f2_score))

        if global_step and global_step % 50 == 0:
            valid_image, valid_label = \
                sess.run(fetches=[valid_image_batch, valid_label_batch])
            valid_accuracy, loss_score, y_pred = sess.run(
                fetches=[metric, loss, tf.nn.sigmoid(logits)],
                feed_dict={x: valid_image, y_: valid_label, keep_prob: 1.0,
                           is_train: False})
            # beta score as specified in competition with beta = 2
            f2_score = metrics.fbeta_score(y_true=valid_label,
                                           y_pred=y_pred > thresholds,
                                           beta=2,
                                           average='samples')
            print("step {0} of {3}, valid accuracy: {1:.4f}, F2 score: {4:.4f}"
                  " log loss: {2:.4f}".format(global_step, valid_accuracy,
                                              loss_score, n, f2_score))


@timeit
@multithreading
def predict(sess, x, keep_prob, is_train, logits, test_image_batch):
    """predict test set using graph previously trained and saved."""
    complete_pred = list()
    while 1:
        try:
            test_image = sess.run(test_image_batch)
            batch_pred = sess.run(tf.nn.sigmoid(logits),
                                  feed_dict={x: test_image, keep_prob: 1.0,
                                             is_train: False})
            complete_pred.append(batch_pred)
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs i.e. 1
            break
    unravelled_array = np.array([array for nested_arrays in complete_pred
                                for array in nested_arrays])
    return unravelled_array


@timeit
def submit(predicted_input, path, tags, thresholds):
    """"produce an output file with predicted probabilities."""
    predictions = predicted_input > thresholds
    tags_predictions = [' '.join(np.array(tags)[boolean_array])
                        for boolean_array in predictions]
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    template = pd.read_csv(
                filepath_or_buffer=path + '/sample_submission.csv',
                encoding='utf8',
                index_col=0)
    df = pd.DataFrame(
                data=tags_predictions,
                columns=template.columns,
                index=template.index)
    df.to_csv(
                path + '/submission_{0}.csv'.format(now),
                encoding='utf8',
                header=True,
                index=True)


@timeit
def restore_session(sess, path):
    """restore hard trained model for predicting."""
    eval_saver = \
        tf.train.import_meta_graph(tf.train.latest_checkpoint(path) + '.meta')
    eval_saver.restore(sess, tf.train.latest_checkpoint(path))
    print('{} restored successfully.'.format(tf.train.latest_checkpoint(path)))


@timeit
def save_session(sess, path, sav):
    """save hard trained model for future predicting."""
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path)
    save_path = sav.save(sess, path + "model_{0}.ckpt".format(now))
    print("Model saved in: {0}".format(save_path))
