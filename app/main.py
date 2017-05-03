# -*- coding: utf-8 -*-

import sys

FETCH = True if 'FETCH' in map(str.upper, sys.argv[1:]) else False
CV_TRAIN = True if 'CV_TRAIN' in map(str.upper, sys.argv[1:]) else False
CV_DETECT = True if 'CV_DETECT' in map(str.upper, sys.argv[1:]) else False
TRAIN = True if 'TRAIN' in map(str.upper, sys.argv[1:]) else False
EVAL = True if 'EVAL' in map(str.upper, sys.argv[1:]) else False


if __name__ == '__main__':
    if FETCH or CV_TRAIN or CV_DETECT:
        from .cv import cv
    elif TRAIN or EVAL:
        from .engine import core
    else:
        raise ValueError('missing mode flags.\n\n'
                         'require one of following:\n'
                         'FETCH, CV_TRAIN, CV_DETECT for computer vsion\n'
                         'TRAIN, EVAL for convolutional neural net.')
