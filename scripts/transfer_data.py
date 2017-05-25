#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transfer_data

this script uploads and downloads image and other data files between Amazon
S3 storage and local
"""
import boto3

import os
import sys
root = os.path.realpath('..')
sys.path.append(root)

from tqdm import tqdm
from app.pipeline import folder_traverse
from app.settings import IMAGE_PATH, EXT, DEFAULT_BUCKET


DOWNLOAD = True if 'DOWNLOAD' in map(str.upper, sys.argv[1:]) else False
UPLOAD = True if 'UPLOAD' in map(str.upper, sys.argv[1:]) else False
ERASE = True if 'ERASE' in map(str.upper, sys.argv[1:]) else False

BUCKETNAME = os.getenv(key='BUCKETNAME', default=DEFAULT_BUCKET)
s3 = boto3.resource('s3')

if UPLOAD:
    fs = folder_traverse(os.path.join(root, IMAGE_PATH + 'train'), ext=EXT)
    local_paths = [os.path.join(directory, filename)
                   for directory in fs for filename in fs[directory]]

    for local_path in tqdm(local_paths, miniters=1):
        remote_key_name = local_path.replace(root + '/./', '')
        s3.meta.client.upload_file(Filename=local_path,
                                   Bucket=BUCKETNAME,
                                   Key=remote_key_name)

elif DOWNLOAD:
    remote_paths = [obj.key for obj in s3.Bucket(BUCKETNAME).objects.all()]
    for remote_path in tqdm(remote_paths, miniters=1):
        local_key_name = os.path.join(root, remote_path)
        if not os.path.exists(os.path.dirname(local_key_name)):
            os.makedirs(os.path.dirname(local_key_name))
        s3.meta.client.download_file(Key=remote_path,
                                     Bucket=BUCKETNAME,
                                     Filename=local_key_name)

if ERASE:
    confirm = input('Delete "{0}"? Type name of the bucket to proceed:'.format(
                    BUCKETNAME))
    if confirm != BUCKETNAME:
        print('Valid confirmation. No action taken.')
    elif confirm == BUCKETNAME:
        s3.Bucket(BUCKETNAME).objects.delete()
        print('All objects in "{0}"'.format(BUCKETNAME))
