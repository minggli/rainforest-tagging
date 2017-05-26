#!/usr/bin/env bash

sudo pip3 install --upgrade pip
sudo pip3 install --upgrade virtualenv
virtualenv amazon -p python3
amazon/bin/pip3 install -r requirements.txt

# so xgboost can be installed via pip on ubuntu but strangely not on Mac
# assuming you have python setup tools and C++ compiler either gcc or clang
if [ "$(uname)" == "Darwin" ]; then
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; cp make/minimum.mk ./config.mk; make -j4
source amazon/bin/activate
cd python-package; sudo python3 setup.py install
cd ..; cd ..

rm -rf xgboost
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
amazon/bin/pip3 install xgboost
fi

# glove vectors and fuller vocabulary
amazon/bin/python3 -m spacy download en_vectors_glove_md
amazon/bin/python3 -m spacy download en_core_web_md
