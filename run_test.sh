#!/bin/bash

# exports need to be put in ./exports subfolder (including xgboost spawning folders)
my_temp="$(pwd)"
export _JAVA_OPTIONS="-Djava.io.tmpdir=$my_temp"
export TMPDIR=$my_temp

# source anaconda and run propriate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate h2o
~/anaconda3/envs/h2o/bin/python ./AutoML_h2o_test_Python.py > log.txt 2>&1

# exit with code 0
exit 0
