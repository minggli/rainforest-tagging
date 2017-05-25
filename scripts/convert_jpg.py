#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_jpg

convert CMYK colourspace to RGB
"""

import os
import sys
root = os.path.realpath('..')
sys.path.append(root)
from PIL import Image
from tqdm import tqdm

from app.pipeline import folder_traverse
from app.settings import IMAGE_PATH

fs = folder_traverse(os.path.join(root, IMAGE_PATH), ext=('.jpg'))

path_to_images = [os.path.join(key, filename) for key in fs
                  for filename in fs[key]]

for path_to_image in tqdm(path_to_images, miniters=1):
    with Image.open(path_to_image) as img:
        if img.mode == 'CMYK':
            img.convert('RGB').save(path_to_image, 'JPEG')
