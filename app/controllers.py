# -*- coding: utf-8 -*-
"""
controllers

handle tensorflow session relating to ConvNet
"""
import os
import time
import tensorflow as tf
import pandas as pd

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
def train(n, sess, x, _y, keep_prob, train_image_batch, train_label_batch,
          valid_image_batch, valid_label_batch, optimiser, metric, loss):
    """train neural network and produce accuracies with validation set."""

    for global_step in range(n):
        train_image, train_label = \
                            sess.run([train_image_batch, train_label_batch])
        optimiser.run(feed_dict={
                            x: train_image, _y: train_label, keep_prob: .5})
        print(global_step, train_label[0])

        if global_step % 10 == 0:
            valid_image, valid_label = \
                sess.run([valid_image_batch, valid_label_batch])
            training_accuracy, loss_score = \
                sess.run([metric, loss], feed_dict={x: valid_image,
                         _y: valid_label, keep_prob: 1.0})
            print("step {0} of {3}, valid accuracy: {1:.4f}, "
                  "log loss: {2:.4f}".format(global_step,
                                             training_accuracy,
                                             loss_score,
                                             n))


@timeit
@multithreading
def predict(sess, x, keep_prob, logits, test_image_batch):
    """predict test set using graph previously trained and saved."""
    complete_probs = list()
    for _ in range(2000):
        try:
            test_image = sess.run(test_image_batch)
            probs = sess.run(tf.nn.softmax(logits),
                             feed_dict={x: test_image, keep_prob: 1.0})
            complete_probs.append(probs)
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs i.e. 1
            break
    unravelled_array = \
        [array for nested_arrays in complete_probs for array in nested_arrays]
    return unravelled_array


@timeit
def submit(complete_probs, path):
    """"produce an output file with predicted probabilities."""
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    template = pd.read_csv(
                filepath_or_buffer=path + 'sample_submission_stg2.csv',
                encoding='utf8',
                index_col=0)
    df = pd.DataFrame(
                data=complete_probs,
                columns=template.columns,
                index=template.index,
                dtype=float).applymap(lambda x: round(x, 6))
    df.to_csv(
                path + 'submission_{0}.csv'.format(now),
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
