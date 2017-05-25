#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_data

this script uploads image and other data files onto Amazon S3 storage
"""
import boto3

import os
import sys
root = os.path.realpath('..')
sys.path.append(root)
from tqdm import tqdm
from app.pipeline import folder_traverse
from app.settings import IMAGE_PATH, EXT, S3_DEFAULT_BUCKET

fs = folder_traverse(os.path.join(root, IMAGE_PATH), ext=EXT)
local_paths = [os.path.join(directory, filename)
               for directory in fs for filename in fs[directory]]

BUCKETNAME = os.getenv(key='BUCKETNAME', default=S3_DEFAULT_BUCKET)
s3_api = boto3.resource('s3')

for local_path in tqdm(local_paths, ministers=1):
    remote_key_name = local_path.replace(root + '/./', '')
    s3_api.meta.client.upload_file(Filename=local_path,
                                   Bucket=BUCKETNAME,
                                   Key=remote_key_name)
# !!! catch bucket not exist error, file overwrite error
