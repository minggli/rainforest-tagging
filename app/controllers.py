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


def calculate_f2_score(y_truth, y_pred, thresholds):
    """batch f2 score as specified in Competition"""
    return metrics.fbeta_score(y_true=y_truth,
                               y_pred=y_pred > thresholds,
                               beta=2,
                               average='samples')


@timeit
@multithreading
def train(n, sess, is_train, pred, label_feed, optimiser, metric, loss,
          thresholds):
    """train neural network and produce accuracies with validation set."""

    for global_step in range(n):

        _, train_accuracy, train_loss, class_prob, train_label = sess.run(
                fetches=[optimiser, metric, loss, pred, label_feed],
                feed_dict={is_train: True})
        f2_score = calculate_f2_score(train_label, class_prob, thresholds)
        print("step {0} of {3}, train accuracy: {1:.4f}, F2 score: {4:.4f}"
              " log loss: {2:.4f}".format(global_step, train_accuracy,
                                          train_loss, n, f2_score))

        if global_step and global_step % 50 == 0:

            valid_accuracy, loss_score, class_prob, valid_label = sess.run(
                fetches=[metric, loss, pred, label_feed])
            f2_score = calculate_f2_score(valid_label, class_prob, thresholds)
            print("step {0} of {3}, valid accuracy: {1:.4f}, F2 score: {4:.4f}"
                  " log loss: {2:.4f}".format(global_step, valid_accuracy,
                                              loss_score, n, f2_score))


@timeit
@multithreading
def predict(sess, pred):
    """predict test set using graph previously trained and saved."""
    complete_pred = list()
    while 1:
        try:
            complete_pred.append(sess.run(pred))
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
    time.sleep(1)
    template = pd.read_csv(
                filepath_or_buffer=path + 'sample/sample_submission.csv',
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
