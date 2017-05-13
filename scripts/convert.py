#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convertjpg

convert CMYK JPG colourspace to RGB JPG
"""

import os
import sys
project_folder = os.path.realpath('.')
sys.path.append(project_folder)
from PIL import Image

from app.pipeline import folder_traverse
from app.settings import IMAGE_PATH

fs = folder_traverse(os.path.join(project_folder, IMAGE_PATH),
                     ext=('.jpg'))
path_to_images = list()

for key in fs:
    for filename in fs[key]:
        path_to_images.append(os.path.join(key, filename))

for path_to_image in path_to_images:
    with Image.open(path_to_image) as img:
        if img.mode == 'CMYK':
            print('converting from CMYK to RGB: {0}'.format(path_to_image))
            img.convert('RGB').save(path_to_image, 'JPEG')
