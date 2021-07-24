#!/bin/bash

# to be runned from the project root dir

SCRIPT_PATH=$(dirname `which $0`)

FILE=data_out/validation_H0.csv
$SCRIPT_PATH/simulation.py --head > $FILE
$SCRIPT_PATH/simulation.py -r 1000 --random-seed 1 >> $FILE
