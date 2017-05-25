#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_tif

convert GeoTiff 4-channel tif images to 4-channel png, just for data storage
and utilise Tensorflow native png decoder as it currently lacks tif decoder.

GeoTiff is to be read by OpenCV
"""

import os
import sys
root = os.path.realpath('..')
sys.path.append(root)
import cv2

from tqdm import tqdm
from app.pipeline import folder_traverse
from app.settings import IMAGE_PATH

fs = folder_traverse(os.path.join(root, IMAGE_PATH),
                     ext=('.tif'))

path_to_images = [os.path.join(key, filename) for key in fs
                  for filename in fs[key]]

for path_to_image in tqdm(path_to_images, miniters=1):
    filename_no_ext = os.path.splitext(path_to_image)[0]
    if not os.path.exists(filename_no_ext + '.png') and \
       os.path.exists(filename_no_ext + '.tif'):
            # print('converting from GeoTiff to PNG: {0}'.format(path_to_image))
            img = cv2.imread(filename_no_ext + '.tif', cv2.IMREAD_UNCHANGED)
            cv2.imwrite(filename_no_ext + '.png', img)
            del img
    if os.path.exists(filename_no_ext + '.png') and \
       os.path.exists(filename_no_ext + '.tif'):
            os.remove(filename_no_ext + '.tif')
