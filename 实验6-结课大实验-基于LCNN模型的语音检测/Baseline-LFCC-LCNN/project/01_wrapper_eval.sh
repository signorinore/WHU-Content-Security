#!/bin/bash
########################
# Script to quickly evaluate a evaluation set
#
# 00_toy_example.sh requires configuration of config.py,
# which further requires preparation of test.lst
# and protocol.txt.
#
# It is convenient when doing numerious experiments
# on the same data set but troublesome when evaluating
# different evaluationg sets.
#
# This script shows one example of quick evaluation using 
# lfcc-lcnn-lstm-sig_toy_example/02_eval_alternative.sh 
#
# It will use DATA/toy_example/eval 
# It will call the evaluat set toy_eval
# It will use __pretrained/trained_network.pt as pre-trained model
# 
# (See Doc of 02_eval_alternative.sh for more details)
#
# Note that we don't need a protocol or meta labels.
# 
########################

# Load python environment
# bash conda.sh

# We will use DATA/toy_example/eval as example
cd DATA
tar -xzf toy_example.tar.gz
cd ..

# Go to the folder
cd baseline_LA

# Run evaluation using pretrained model 
#  bash 02_eval_alternative.sh PATH_TO_WAV_DIR NAME_OF_DATA_SET TRAINED_MODEL
# For details, please check 02_eval_alternative.sh 
bash 02_eval_alternative.sh $PWD/../DATA/toy_example/eval toy_eval __pretrained/trained_network.pt
