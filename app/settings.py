# -*- coding: utf-8 -*-

import numpy as np

# Convolutional Neural Network
MODEL_PATH = './trained_models/'
IMAGE_PATH = './data/'
# IMAGE_PATH = './data/original_images/'
IMAGE_SHAPE = (256, 256, 3)
BATCH_SIZE = 64
MAX_STEPS = 5000
ALPHA = 1e-2
BETA = 1e-2

TAGS = np.array(
       ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
        'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
        'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
        'selective_logging', 'slash_burn', 'water'])

TAGS_WEIGHTINGS = [0.8941, 0.9971, 0.9926, 0.9971, 0.9991, 0.7579, 0.98,
                   0.9991, 0.961, 0.9686, 0.9769, 0.9378, 0.6751, 0.9307,
                   0.9971, 0.9982, 0.9377]
