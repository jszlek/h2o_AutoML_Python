#!/bin/bash

# clean ./export folder before modeling - avoid cross-calculations
for f in ./export/*
    do rm -Rf "$f"
done

# exports need to be put in ./exports subfolder (including xgboost spawning folders)
my_temp="$(pwd)/export"
export _JAVA_OPTIONS="-Djava.io.tmpdir=$my_temp"
export TMPDIR=$my_temp

# source anaconda and run propriate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate h2o3
~/anaconda3/envs/h2o3/bin/python ./h2o_Python_v16.py > log.txt 2>&1

# clean ./export folder after modeling
for f in ./export/*
    do rm -Rf "$f"
done

# exit with code 0
exit 0
