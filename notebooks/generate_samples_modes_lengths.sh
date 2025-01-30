#! /usr/bin/env bash

MAX_PROC=$1
NUMBER_OF_SAMPLES=$2

# We want to generate the data in the working directory; link the shell scripts here.
ln -s ../shell_utilities/run_experiment_families.py
ln -s ../shell_utilities/complexity_experiment.py


# Memory complexity of l=(1,l,l^2) as function of l=1,...,6 (for value 6 see below, here iterate only until l=5) and m=2,22,...,282
python3 run_experiment_families.py \
        -E "-m m -ell 1,l,ll -rep $NUMBER_OF_SAMPLES -no-N -r m=2:283:20 -r l=1:6 -e ll=l*l -- -wN -LP -Nh paths -p" \
        -P $MAX_PROC


# Memory complexity of l=(1,6,36) as function of the number of modes m=2,3...,44; 64,84,...,284
# Note we have more granular data (in terms of the values of m used) for this loop system, so we run it separately
python3 run_experiment_families.py \
        -E "-m m -ell 1,6,36 -rep $NUMBER_OF_SAMPLES -no-N -r m=2:44 -- -wN -LP -Nh paths -p" \
        -E "-m m -ell 1,6,36 -rep $NUMBER_OF_SAMPLES -no-N -r m=44:285:20 -- -wN -LP -Nh paths -p" \
        -P $MAX_PROC



# Remove the symlinks.
rm ./run_experiment_families.py
rm ./complexity_experiment.py
