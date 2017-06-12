# -*- coding: utf-8 -*-
"""
metadata

gradient boosting trees on meta data of images (e.g. mean, kurtosis, skewness)
"""

import xgboost as xgb
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from ..settings import (IMAGE_PATH, IMAGE_SHAPE, TAGS, TAGS_THRESHOLDS,
                        BATCH_SIZE, VALID_SIZE, EXT, OUTPUT_PATH, N_THREADS)
from ..pipeline import data_pipe, generate_data_skeleton, multithreading
from ..controllers import calculate_f2_score, submit


def extract_meta_features(batch_tensor):
    """extract metadata information from expected 4-D image tensor
    (batch_size, width, height, dimension)"""
    # per image mean per channel resulting in (batch_size, dimension)
    mean, variance = tf.nn.moments(batch_tensor, axes=(1, 2))
    std = tf.sqrt(variance)
    channel_max = tf.reduce_max(batch_tensor, axis=(1, 2))
    channel_min = tf.reduce_min(batch_tensor, axis=(1, 2))
    mean_sub = batch_tensor - tf.reshape(mean, shape=[-1, 1, 1, 4])
    skewness = tf.reduce_mean(mean_sub ** 3, axis=(1, 2)) / std ** 3
    kurtosis = tf.reduce_mean(mean_sub ** 4, axis=(1, 2)) / std ** 4
    feature_stack = tf.stack(values=[mean, std, channel_max, channel_min,
                             skewness, kurtosis],
                             axis=1)
    # output stacked features in shape of [batch_size, 6, dimension]
    return feature_stack


@multithreading
def materialise_data(meta_batch, label_batch, verbose=True):
    """simply aggregate data for Gradient Boosting Tree classifier"""
    feats, labels, count = list(), list(), 0
    while True:
        try:
            meta, label = sess.run(fetches=[meta_batch, label_batch])
            feats.append(meta)
            labels.append(label)
            count += 1
            if verbose:
                print('mini-batches loaded: {0}'.format(count))
        except tf.errors.OutOfRangeError as e:
            # pipe exhausted with pre-determined number of epochs i.e. 1
            break
        meta_flat = np.array([array for nested in feats for array in nested])
        label_flat = np.array([array for nested in labels for array in nested])
    return meta_flat.reshape([-1, 6 * IMAGE_SHAPE[-1]]), label_flat


train_file_array, train_label_array, valid_file_array, valid_label_array = \
                                            generate_data_skeleton(
                                                root_dir=IMAGE_PATH + 'train',
                                                valid_size=VALID_SIZE,
                                                ext=EXT)
train_image_batch, train_label_batch = data_pipe(
                                                train_file_array,
                                                train_label_array,
                                                num_epochs=1,
                                                shape=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                augmentation=True,
                                                shuffle=True,
                                                threads=N_THREADS)
train_meta_batch = extract_meta_features(train_image_batch)

valid_image_batch, valid_label_batch = data_pipe(
                                                valid_file_array,
                                                valid_label_array,
                                                num_epochs=1,
                                                shape=IMAGE_SHAPE,
                                                batch_size=BATCH_SIZE,
                                                augmentation=True,
                                                shuffle=True,
                                                threads=N_THREADS)
valid_meta_batch = extract_meta_features(valid_image_batch)

init = tf.local_variables_initializer()

sess = tf.Session()
sess.run(init)

with sess:
    X_valid, y_valid = materialise_data(valid_meta_batch, valid_label_batch)

sess = tf.Session()
sess.run(init)

with sess:
    X_train, y_train = materialise_data(train_meta_batch, train_label_batch)

# OvR for multi-label classification
y_valid_pred = np.zeros_like(y_valid, dtype=np.float32)
models = list()
for label_index in tqdm(range(17), miniters=1):
    clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=300)
    clf.fit(X_train, y_train[:, label_index])
    # assign position probability to index location
    np.copyto(y_valid_pred[:, label_index], clf.predict_proba(X_valid)[:, 1])
    models.append(clf)

print('validation F2-score: {0}'.format(
        calculate_f2_score(y_valid, y_valid_pred, TAGS_THRESHOLDS)))

tf.reset_default_graph()
test_file_array, test_label_sample = generate_data_skeleton(
                                    root_dir=IMAGE_PATH + 'test',
                                    valid_size=None,
                                    ext=EXT)
test_image_batch, test_ph_batch = data_pipe(
                                    test_file_array,
                                    test_label_sample,
                                    num_epochs=1,
                                    shape=IMAGE_SHAPE,
                                    batch_size=BATCH_SIZE,
                                    augmentation=False,
                                    shuffle=False)
test_meta_batch = extract_meta_features(test_image_batch)

sess = tf.Session()
sess.run(tf.local_variables_initializer())

with sess:
    X_test, y_ph = materialise_data(test_meta_batch, test_ph_batch)

y_pred = np.zeros(shape=[X_test.shape[0], 17], dtype=np.float32)
for label_index in tqdm(range(17), miniters=1):
    clf = models[label_index]
    np.copyto(y_pred[:, label_index], clf.predict_proba(X_test)[:, 1])
xgb_prob = y_pred
submit(xgb_prob, OUTPUT_PATH, TAGS, TAGS_THRESHOLDS)

# removing references to manually free up memory
tf.reset_default_graph()
del sess, models, clf
