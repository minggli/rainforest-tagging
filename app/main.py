# -*- coding: utf-8 -*-

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


TRAIN = True if 'TRAIN' in map(str.upper, sys.argv[1:]) else False
EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False

if 'ENSEMBLE' in map(str.upper, sys.argv[1:]):
    ENSEMBLE = 5
    TRAIN = EVAL = True
else:
    ENSEMBLE = 1

if __name__ == '__main__':
    if not any([TRAIN, EVAL, ENSEMBLE]):
        raise ValueError('missing mode flags.\n\n'
                         'require one of following:\n'
                         'TRAIN, EVAL for convolutional neural net.')
    elif TRAIN or EVAL:
        from .engine import ovr
