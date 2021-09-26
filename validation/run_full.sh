#!/bin/bash

DIR=`dirname $0`

####################### all cont
echo "normal y, yhat, c"
python3 $DIR/simulation.py --out_prefix=ccc_full_normal --mode=full

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1  --mode=full

echo "non-nomral y,c,yhat: delta=1.05, epsilon=3 (kurt: 2 skew: -1.5)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-all2 --delta-yc=1.05 --epsilon-yc=3 --delta-yhat=1.05 --epsilon-yhat=3 --mode=full

echo "non-nomral y,c,yhat: delta=1.5, epsilon=5 (kurt: 5 skew: -2)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-all3 --delta-yc=1.5 --epsilon-yc=5 --delta-yhat=1.5 --epsilon-yhat=5 --mode=full

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
python3 $DIR/simulation.py --out_prefix=ccc_full_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0  --mode=full

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3  --mode=full

####################### all cont - squared relationship
echo "normal y, yhat, c"
python3 $DIR/simulation.py --out_prefix=ccc_full_normal --mode=full

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1  --mode=full

echo "non-nomral y,c,yhat: delta=1.05, epsilon=3 (kurt: 2 skew: -1.5)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-all2 --delta-yc=1.05 --epsilon-yc=3 --delta-yhat=1.05 --epsilon-yhat=3 --mode=full

echo "non-nomral y,c,yhat: delta=1.5, epsilon=5 (kurt: 5 skew: -2)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-all3 --delta-yc=1.5 --epsilon-yc=5 --delta-yhat=1.5 --epsilon-yhat=5 --mode=full

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
python3 $DIR/simulation.py --out_prefix=ccc_full_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0  --mode=full

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
python3 $DIR/simulation.py --out_prefix=ccc_full_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3  --mode=full


####################### binary c

echo "ccb normal y, yhat, c"
python3 $DIR/simulation.py --out_prefix=ccb_full_normal --cat-c  --mode=full

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
python3 $DIR/simulation.py --out_prefix=ccb_full_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-c  --mode=full

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
python3 $DIR/simulation.py --out_prefix=ccb_full_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-c  --mode=full

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
python3 $DIR/simulation.py --out_prefix=ccb_full_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-c  --mode=full

####################### binary y, yhat

echo "bbc normal y, yhat, c"
python3 $DIR/simulation.py --out_prefix=bbc_full_normal --cat-c --cat-yyhat  --mode=full

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
python3 $DIR/simulation.py --out_prefix=bbc_full_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-yyhat  --mode=full

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
python3 $DIR/simulation.py --out_prefix=bbc_full_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-yyhat  --mode=full

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
python3 $DIR/simulation.py --out_prefix=bbc_full_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-yyhat  --mode=full


####################### binary all

echo "bbb normal y, yhat, c"
python3 $DIR/simulation.py --out_prefix=bbb_full_normal --cat-c --cat-yyhat  --mode=full

echo "non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)"
python3 $DIR/simulation.py --out_prefix=bbb_full_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-yyhat  --mode=full

echo "y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)"
python3 $DIR/simulation.py --out_prefix=bbb_full_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-yyhat  --mode=full

echo "normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)"
python3 $DIR/simulation.py --out_prefix=bbb_full_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-yyhat  --mode=full
