#!/bin/bash

DIR=`dirname $0`
#echo python3 $DIR

####################### all cont
echo "normal y, yhat, c"
python3 $DIR/simulation.py --out_prefix=pearson_ccc_partial_normal

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
python3 $DIR/simulation.py --out_prefix=pearson_ccc_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --mode='partial_pearson'

echo "non-nomral y,c,yhat: delta=1.05, epsilon=3 (kurt: 2 skew: -1.5)"
python3 $DIR/simulation.py --out_prefix=pearson_ccc_partial_non-normal-all2 --delta-yc=1.05 --epsilon-yc=3 --delta-yhat=1.05 --epsilon-yhat=3 --mode='partial_pearson'

echo "non-nomral y,c,yhat: delta=1.5, epsilon=5 (kurt: 5 skew: -2)"
python3 $DIR/simulation.py --out_prefix=pearson_ccc_partial_non-normal-all3 --delta-yc=1.5 --epsilon-yc=5 --delta-yhat=1.5 --epsilon-yhat=5 --mode='partial_pearson'

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
python3 $DIR/simulation.py --out_prefix=pearson_ccc_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --mode='partial_pearson'

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
python3 $DIR/simulation.py --out_prefix=pearson_ccc_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --mode='partial_pearson'
