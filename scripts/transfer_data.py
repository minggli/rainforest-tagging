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
from multiprocessing import Pool
from app.pipeline import folder_traverse
from app.settings import IMAGE_PATH, EXT, DEFAULT_BUCKET


DOWNLOAD = True if 'DOWNLOAD' in map(str.upper, sys.argv[1:]) else False
UPLOAD = True if 'UPLOAD' in map(str.upper, sys.argv[1:]) else False
ERASE = True if 'ERASE' in map(str.upper, sys.argv[1:]) else False

BUCKETNAME = os.getenv(key='BUCKETNAME', default=DEFAULT_BUCKET)
s3 = boto3.resource('s3')


def upload_to_remote(local_path):
    """upload data onto S3"""
    remote_key_name = local_path.replace(root + '/./', '')
    s3.meta.client.upload_file(Filename=local_path,
                               Bucket=BUCKETNAME,
                               Key=remote_key_name)


def download_from_remote(remote_path):
    """create local paths and download from Amazon S3"""
    local_key_name = os.path.join(root, remote_path)
    if not os.path.exists(os.path.dirname(local_key_name)):
        try:
            os.makedirs(os.path.dirname(local_key_name))
        except FileExistsError:
            pass
    s3.meta.client.download_file(Key=remote_path,
                                 Bucket=BUCKETNAME,
                                 Filename=local_key_name)


if UPLOAD:
    p = Pool(8)
    fs = folder_traverse(os.path.join(root, IMAGE_PATH), ext=EXT)
    remote_paths = [obj.key for obj in s3.Bucket(BUCKETNAME).objects.all()]
    local_paths = [os.path.join(directory, filename) for directory in fs
                   for filename in fs[directory] if
                   os.path.join(directory, filename).replace(root + '/./', '')
                   not in remote_paths]
    for _ in tqdm(p.imap_unordered(upload_to_remote, local_paths),
                  total=len(local_paths)):
        pass


elif DOWNLOAD:
    p = Pool(8)
    remote_paths = [obj.key for obj in s3.Bucket(BUCKETNAME).objects.all()]
    filtered_remote_path = [key for key in remote_paths if key.endswith(EXT)]
    for _ in tqdm(p.imap_unordered(download_from_remote, filtered_remote_path),
                  total=len(filtered_remote_path)):
        pass

elif ERASE:
    confirm = input('Delete "{0}"? Type name of the bucket to proceed:'.format(
                    BUCKETNAME))
    if confirm != BUCKETNAME:
        print('Invalid confirmation. No action taken.')
    elif confirm == BUCKETNAME:
        s3.Bucket(BUCKETNAME).objects.delete()
        print('All objects in "{0}" deleted.'.format(BUCKETNAME))
