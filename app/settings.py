# -*- coding: utf-8 -*-

# Convolutional Neural Network
MODEL_PATH = './trained_models/'
IMAGE_PATH = './data/'
OUTPUT_PATH = './output/'
DEFAULT_BUCKET = 'rainforest-satellite-images'
EXT = ('.jpg', '.csv')
IMAGE_SHAPE = (128, 128, 3)
BATCH_SIZE = 96
N_THREADS = 4
# roughly 10 epochs of training data
MAX_STEPS = 5000
EPOCHS = 5
ALPHA = 4e-5
BETA = 1e-2
VALID_SIZE = 0.01
KEEP_RATE = .8

# Ioffe and Szegedy 2015 suggests 5x learning rate with batch normalization
# ALPHA *= 5

# with data augmentation turned on, need long training steps
AUGMENT = True
if AUGMENT:
    MAX_STEPS *= 2

TAGS = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
        'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
        'habitation', 'haze', 'partly_cloudy', 'primary', 'road',
        'selective_logging', 'slash_burn', 'water']

TAGS_WEIGHTINGS = [0.8941, 0.9971, 0.9926, 0.9971, 0.9991, 0.7579, 0.98,
                   0.9991, 0.961, 0.9686, 0.9769, 0.9378, 0.6751, 0.9307,
                   0.9971, 0.9982, 0.9377]

# TAGS_THRESHOLDS = [0.3, 0.2, 0.2, 0.2, 0.1, 0.4, 0.2,
#                    0.1, 0.2, 0.2, 0.2, 0.2, 0.5, 0.2,
#                    0.2, 0.1, 0.2]
#
TAGS_THRESHOLDS = [0.245, 0.1375, 0.2225, 0.19, 0.0475, 0.2375, 0.12,
                   0.0875, 0.265, 0.2175, 0.1925, 0.1625, 0.2625, 0.21,
                   0.14, 0.085, 0.205]
#
# TAGS_THRESHOLDS = [0.25, 0.33, 0.17, 0.04, 0.07, 0.23, 0.16, 0.2, 0.23, 0.19,
#                    0.24, 0.14, 0.22, 0.24, 0.12, 0.1, 0.26]
