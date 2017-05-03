# -*- coding: utf-8 -*-

import numpy as np

# Convolutional Neural Network
MODEL_PATH = './trained_models/'
IMAGE_PATH = './data/'
# IMAGE_PATH = './data/original_images/'
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 100
MAX_STEPS = 50
ALPHA = 1e-3
BETA = 1e-2

TAGS = np.array(
       ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
        'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
        'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
        'selective_logging', 'slash_burn', 'water'])
