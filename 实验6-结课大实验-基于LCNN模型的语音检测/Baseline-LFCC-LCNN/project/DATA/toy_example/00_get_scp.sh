#!/bin/bash

# Just list the trial names in separate files
# The name convention is based on ASVspoof2019

# train set
mkdir scp
grep LA_T protocol.txt | awk '{print $2}' > scp/train.lst
# validation set
grep LA_D protocol.txt | awk '{print $2}' > scp/val.lst
# evaluation set
grep LA_E protocol.txt | awk '{print $2}' > scp/test.lst
