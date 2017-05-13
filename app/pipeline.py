# -*- coding: utf-8 -*-
"""
pipeline

data pipeline from image root folder to processed tensors of train test batches
for images and labels
"""
import os
import functools
import collections

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import model_selection, preprocessing


def folder_traverse(root_dir, ext=('.jpg')):
    """recusively map all image files from root directory"""

    if not os.path.exists(root_dir):
        raise RuntimeError('{0} doesn\'t exist.'.format(root_dir))

    file_structure = collections.defaultdict(list)
    for item in os.scandir(root_dir):
        if item.is_dir():
            file_structure.update(folder_traverse(item.path, ext))
        elif item.is_file() and item.name.endswith(ext):
            file_structure[os.path.dirname(item.path)].append(item.name)
    return file_structure


def resample(feature_index, labels, balance='auto'):
    """use oversampling to balance class, after split of training set."""

    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(ratio=balance)
    feature_index = np.array(feature_index).reshape(-1, 1)
    resampled_index, _ = ros.fit_sample(feature_index, labels)
    resampled_index = [i for nested in resampled_index for i in nested]
    return resampled_index


def generate_data_skeleton(root_dir,
                           ext=('.jpg', '.csv'),
                           valid_size=None,
                           oversample=False):
    """turn file structure into human-readable pandas dataframe"""
    file_structure = folder_traverse(root_dir, ext=ext)
    reversed_fs = {k + '/' + f: os.path.splitext(f)[0]
                   for k, v in file_structure.items() for f in v}

    # find the first csv and load it in memory and remove it from dictionary
    for key in reversed_fs:
        if key.endswith('.csv'):
            df_csv = pd.read_csv(key, dtype=np.str)
            reversed_fs.pop(key)
            break

    df = pd.DataFrame.from_dict(data=reversed_fs, orient='index').reset_index()
    df.rename(columns={'index': 'path_to_file', 0: 'filename'}, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df = df_csv.merge(right=df,
                      how='left',
                      left_on='image_name',
                      right_on='filename').dropna(axis=0)

    discrete_labels = [string.split(' ') for string in df['tags'].tolist()]
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit(discrete_labels)

    X = np.array(df['path_to_file'])
    y = mlb.transform(discrete_labels)
    X_codified = df['path_to_file'].index
    y_codified = pd.Categorical(df['tags']).codes

    if valid_size:
        print('tags one-hot encoded: \n{0}'.format(mlb.classes_))

        X_train_codified, X_valid_codified, y_train_codified,\
            y_valid_codified = model_selection.train_test_split(
                                X_codified,
                                y_codified,
                                test_size=valid_size)

        if oversample:
            resampled_train_idx = resample(X_train_codified, y_train_codified)
            resampled_valid_idx = resample(X_valid_codified, y_valid_codified)

            X_train, y_train = X[resampled_train_idx], y[resampled_train_idx]
            X_valid, y_valid = X[resampled_valid_idx], y[resampled_valid_idx]
            print('To balance classes, training data has been oversampled'
                  ' to: {0}'.format(len(resampled_train_idx) +
                                    len(resampled_valid_idx)))
        elif not oversample:
            X_train, y_train = X[X_train_codified], y[X_train_codified]
            X_valid, y_valid = X[X_valid_codified], y[X_valid_codified]

        print('training: {0} samples; validation: {1} samples.'.format(
                                        X_train.shape[0], X_valid.shape[0]))
        return X_train, y_train, X_valid, y_valid
    elif not valid_size:
        print('test: {0} samples.'.format(X.shape[0]))
        return X, y


def make_queue(paths_to_image, labels, num_epochs=None, shuffle=True):
    """returns an Ops Tensor with queued image and label pair"""
    images = tf.convert_to_tensor(paths_to_image, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.uint8)
    input_queue = tf.train.slice_input_producer(
                                tensor_list=[images, labels],
                                num_epochs=num_epochs,
                                shuffle=shuffle)
    return input_queue


def decode_transform(input_queue,
                     shape=None,
                     standardize=True,
                     distortion=False):
    """a single decode and transform function that applies standardization with
    mean centralisation."""
    # input_queue allows slicing with 0: path_to_image, 1: encoded label
    label_queue = input_queue[1]
    image_queue = tf.read_file(input_queue[0])

    # !!! decode_jpeg only accepts RGB jpg but not raising error for CMYK :(
    original_image = tf.image.decode_image(image_queue, channels=0)

    # crop larger images to 256*256, this func doesn't 'resize'.
    cropped_img = tf.image.resize_image_with_crop_or_pad(
                                image=original_image,
                                target_height=256,
                                target_width=256)

    # resize cropped images to desired shape
    img = tf.image.resize_images(images=cropped_img, size=[shape[0], shape[1]])
    img.set_shape(shape)

    if distortion:
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, lower=.2, upper=2)
    # apply standardization
    if standardize:
        img = tf.image.per_image_standardization(img)

    return img, label_queue


def batch_generator(image, label, batch_size=None, shuffle=True):
    """turn data queue into batches"""
    if shuffle:
        return tf.train.shuffle_batch(
                                tensors=[image, label],
                                batch_size=batch_size,
                                num_threads=4,
                                capacity=1e3,
                                min_after_dequeue=200,
                                allow_smaller_final_batch=True)
    elif not shuffle:
        return tf.train.batch(
                                tensors=[image, label],
                                batch_size=batch_size,
                                num_threads=1,
                                # thread number must be one to be unshuffled.
                                capacity=1e3,
                                allow_smaller_final_batch=True)


def data_pipe(paths_to_image,
              labels,
              num_epochs=None,
              batch_size=None,
              shape=None,
              shuffle=True):
    """so one-in-all from data directory to iterated data feed in batches"""
    resized_image_queue, label_queue = decode_transform(make_queue(
                                paths_to_image,
                                labels,
                                num_epochs=num_epochs,
                                shuffle=shuffle),
                                shape=shape)
    image_batch, label_batch = batch_generator(
                                resized_image_queue,
                                label_queue,
                                batch_size=batch_size,
                                shuffle=shuffle)
    return image_batch, label_batch


def multithreading(func):
    """decorator using tensorflow threading ability."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        func_output = func(*args, **kwargs)
        try:
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)
        except (tf.errors.CancelledError, RuntimeError) as e:
            pass
        return func_output
    return wrapper
