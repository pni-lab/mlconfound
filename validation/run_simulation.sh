#!/bin/bash

SCRIPT_PATH=$(dirname `which $0`)

FILE=test_H0.csv
$SCRIPT_PATH/simulation.py --head > $FILE
$SCRIPT_PATH/simulation.py -r 100 >> $FILE
