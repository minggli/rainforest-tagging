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


def generate_data_skeleton(root_dir, ext=('.jpg', '.csv'), valid_size=None):
    """turn file structure into human-readable pandas dataframe"""
    file_structure = folder_traverse(root_dir, ext=ext)
    reversed_fs = {k + '/' + f: os.path.splitext(f)[0]
                   for k, v in file_structure.items() for f in v}

    for key in reversed_fs:
        if key.endswith('.csv'):
            df_csv = pd.read_csv(key)
            reversed_fs.pop(key)
            break

    df = pd.DataFrame.from_dict(data=reversed_fs, orient='index').reset_index()
    df.rename(columns={'index': 'path_to_file', 0: 'filename'}, inplace=True)
    df.reset_index(inplace=True, drop=True)

    df = df_csv.merge(right=df,
                      how='left',
                      left_on='image_name',
                      right_on='filename')

    train_labels = [string.split(' ') for string in df['tags'].tolist()]
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit(train_labels)
    X = np.array(df['path_to_file'])
    y = mlb.transform(train_labels)
    if valid_size:
        print('tags one-hot encoded: \n{0}'.format(mlb.classes_))
        X_train, X_valid, y_train, y_valid = model_selection.train_test_split(
            X, y, test_size=valid_size, stratify=y)
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


def decode_transform(input_queue, shape=None, standardize=True, distort=True):
    """a single decode and transform function that applies standardization with
    mean centralisation."""
    # input_queue allows slicing with 0: path_to_image, 1: encoded label
    label_queue = input_queue[1]
    image_queue = tf.read_file(input_queue[0])
    original_img = tf.image.decode_jpeg(image_queue, channels=shape[2])

    # crop larger images to 256*256, this func doesn't 'resize'.
    cropped_img = tf.image.resize_image_with_crop_or_pad(
                                image=original_img,
                                target_height=256,
                                target_width=256)

    # resize cropped images to desired shape
    resized_img = tf.image.resize_images(
                                images=cropped_img,
                                size=[shape[0], shape[1]])
    resized_img.set_shape(shape)
    img = resized_img

    if distort:
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
