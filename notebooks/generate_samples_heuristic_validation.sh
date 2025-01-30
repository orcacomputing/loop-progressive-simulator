#! /usr/bin/env bash

MAX_PROC=$1
NUMBER_OF_SAMPLES=$2

# We want to generate the data in the working directory; link the shell scripts here.
ln -s ../shell_utilities/run_experiment_families.py
ln -s ../shell_utilities/complexity_experiment.py


# Validation of the heuristic. First, generate the real data:
python3 run_experiment_families.py \
        -E "-m 12 -ell 1,2,4 -rep $NUMBER_OF_SAMPLES -no-N -- -wA -thr -p" \
        -E "-m 10 -ell 1,2,3 -rep $NUMBER_OF_SAMPLES -no-N -- -wA -thr -p" \
        -E "-m 10 -ell 1,4 -rep $NUMBER_OF_SAMPLES -no-N -- -wA -thr -p" \
        -P $MAX_PROC # use this to limit the number of available processors

# The following have been removed because they take too long; add them back if you have a powerful computer or enough time
        # -E "-m 22 -ell 1,7,9 -rep $NUMBER_OF_SAMPLES -no-N -- -wA -thr -p" \
        # -E "-m 18 -ell 1,6,7 -rep $NUMBER_OF_SAMPLES -no-N -- -wA -thr -p" \
        # -E "-m 15 -ell 1,9 -rep $NUMBER_OF_SAMPLES -no-N -- -wA -thr -p" \



# Next, the heuristic data:
python3 run_experiment_families.py \
        -E "-m 12 -ell 1,2,4 -rep $NUMBER_OF_SAMPLES -no-N -- -wN -LP -Nh paths -p" \
        -E "-m 10 -ell 1,2,3 -rep $NUMBER_OF_SAMPLES -no-N -- -wN -LP -Nh paths -p" \
        -E "-m 10 -ell 1,4 -rep $NUMBER_OF_SAMPLES -no-N -- -wN -LP -Nh paths -p" \
        -P $MAX_PROC

# If you add the above removed experiments, add here their heuristic counterparts:
        # -E "-m 22 -ell 1,7,9 -rep $NUMBER_OF_SAMPLES -no-N -- -wN -LP -Nh paths -p" \
        # -E "-m 18 -ell 1,6,7 -rep $NUMBER_OF_SAMPLES -no-N -- -wN -LP -Nh paths -p" \
        # -E "-m 15 -ell 1,9 -rep $NUMBER_OF_SAMPLES -no-N -- -wN -LP -Nh paths -p" \


# Remove the symbolic links
rm ./run_experiment_families.py
rm ./complexity_experiment.py
