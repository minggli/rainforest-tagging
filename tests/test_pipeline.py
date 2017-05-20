#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline

to see if data flows through app.pipeline module as expected.
"""

import os
import sys
project_folder = os.path.realpath('..')
sys.path.append(project_folder)
import tensorflow as tf
import numpy as np

from PIL import Image
from app.pipeline import generate_data_skeleton, data_pipe, multithreading
from app.settings import IMAGE_PATH, BATCH_SIZE, IMAGE_SHAPE


train_file_array, train_label_array, valid_file_array, valid_label_array =\
    generate_data_skeleton(root_dir=os.path.join(project_folder,
                                                 IMAGE_PATH,
                                                 'train'),
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
test_file_array, _ = generate_data_skeleton(
                                    root_dir=os.path.join(project_folder,
                                                          IMAGE_PATH + 'test'),
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

sess = tf.Session()
init_op = tf.group(tf.local_variables_initializer(),
                   tf.global_variables_initializer())
sess.run(init_op)


@multithreading
def test_shuffle_queue():
    whole_train_images = list()
    for _ in range(3):
        image_batch = valid_image_batch.eval()
        whole_train_images.append(image_batch)
    return [piece for blk in whole_train_images for piece in blk]


@multithreading
def test_unshuffle_queue():
    whole_test_images = list()
    while True:
        try:
            test_image = sess.run(test_image_batch)
            whole_test_images.append(test_image)
            print(test_image.shape)
        except tf.errors.OutOfRangeError as e:
            break
    return [piece for blk in whole_test_images for piece in blk]


with sess:
    total = test_shuffle_queue()
    n = int(input('choose a image to test'))
    print(total[n])
    # Image.fromarray(np.array(total[n], dtype=np.uint8)).show()
