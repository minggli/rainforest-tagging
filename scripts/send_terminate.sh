#!/usr/bin/env bash

export SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $SCRIPT_DIR

aws s3 sync ../output s3://rainforest-save/output
aws s3 sync ../trained_models s3://rainforest-save/trained_models

sudo shutdown -h now
