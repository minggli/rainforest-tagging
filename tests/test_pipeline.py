#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline

to see if data flows through app.pipeline module as expected.
"""

import os
import sys
sys.path.append(os.path.realpath('.'))
import tensorflow as tf
import numpy as np

from PIL import Image

from app.pipeline import generate_data_skeleton, data_pipe, multithreading
from app.settings import IMAGE_PATH, BATCH_SIZE, IMAGE_SHAPE


train_file_array, train_label_array, valid_file_array, valid_label_array =\
    generate_data_skeleton(root_dir=IMAGE_PATH + 'train', valid_size=.15,
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
sess = tf.Session()
init_op = tf.group(tf.local_variables_initializer(),
                   tf.global_variables_initializer())
sess.run(init_op)


@multithreading
def test_queue():
    whole_test_images = list()
    for _ in range(3):
        image_batch = valid_image_batch.eval(session=sess)
        whole_test_images.append(image_batch)
    return [piece for blk in whole_test_images for piece in blk]


with sess:
    total = test_queue()
    n = int(input('choose a image to test'))
    print(total[n])
    Image.fromarray(np.array(total[n], dtype=np.uint8)).show()
