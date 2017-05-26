#!/usr/bin/env bash

aws s3 sync ../output s3://rainforest-outputs
shutdown -h now
