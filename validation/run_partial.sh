#!/bin/bash

DIR=`dirname $0`

####################### all cont
echo "normal y, yhat, c"
$DIR/validation/simulation.py --out_prefix=ccc_partial_normal

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
$DIR/validation/simulation.py --out_prefix=ccc_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
$DIR/validation/simulation.py --out_prefix=ccc_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
$DIR/validation/simulation.py --out_prefix=ccc_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3

####################### binary c

echo "normal y, yhat, c"
$DIR/validation/simulation.py --out_prefix=ccb_partial_normal --cat-c

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
$DIR/validation/simulation.py --out_prefix=ccb_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-c

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
$DIR/validation/simulation.py --out_prefix=ccb_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-c

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
$DIR/validation/simulation.py --out_prefix=ccb_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-c

####################### binary y, yhat

echo "normal y, yhat, c"
$DIR/validation/simulation.py --out_prefix=bbc_partial_normal --cat-c --cat-yyhat

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
$DIR/validation/simulation.py --out_prefix=bbc_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-yyhat

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
$DIR/validation/simulation.py --out_prefix=bbc_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-yyhat

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
$DIR/validation/simulation.py --out_prefix=bbc_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-yyhat


####################### binary all

echo "normal y, yhat, c"
$DIR/validation/simulation.py --out_prefix=bbb_partial_normal --cat-c --cat-yyhat

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
$DIR/validation/simulation.py --out_prefix=bbb_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-yyhat

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
$DIR/validation/simulation.py --out_prefix=bbb_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-yyhat

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
$DIR/validation/simulation.py --out_prefix=bbb_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-yyhat