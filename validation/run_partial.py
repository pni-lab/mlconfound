import os

dir = os.getcwd()
#print(python3 $DIR)

####################### all cont
print("normal y, yhat, c ")
os.system("python3 " + dir + "/simulation.py /simulation.py --out_prefix=ccc_partial_normal")

####################### all cont - squared relationship
print("normal y, yhat, c - squared")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_normal --nonlin-trf=squared")

#######################

print("non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1")

print("non-nomral y,c,yhat: delta=1.05, epsilon=3 (kurt: 2 skew: -1.5)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_non-normal-all2 --delta-yc=1.05 --epsilon-yc=3 --delta-yhat=1.05 --epsilon-yhat=3")

print("non-nomral y,c,yhat: delta=1.5, epsilon=5 (kurt: 5 skew: -2)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_non-normal-all3 --delta-yc=1.5 --epsilon-yc=5 --delta-yhat=1.5 --epsilon-yhat=5")

print("y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0")

print("normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3")

print("non-nomral y,c,yhat: delta=5, epsilon=10")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_non-normal-all4 --delta-yc=5 --epsilon-yc=10 --delta-yhat=5 --epsilon-yhat=10")

print("non-nomral y,c,yhat: delta=5, epsilon=-10")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccc_partial_non-normal-all5 --delta-yc=5 --epsilon-yc=-10 --delta-yhat=5 --epsilon-yhat=-10")

####################### binary c

print("ccb normal y, yhat, c")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccb_partial_normal --cat-c")

print("non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccb_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-c")

print("y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccb_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-c")

print("normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)")
os.system("python3 " + dir + "/simulation.py --out_prefix=ccb_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-c")

####################### binary y, yhat

print("bbc normal y, yhat, c")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbc_partial_normal --cat-c --cat-yyhat")

print("non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbc_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-yyhat")

print("y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbc_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-yyhat")

print("normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbc_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-yyhat")


####################### binary all

print("bbb normal y, yhat, c")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbb_partial_normal --cat-c --cat-yyhat")

print("non-nomral y,c,yhat: delta=1, epsilon=1 (kurt: 1.1 skew: -1.1)")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbb_partial_non-normal-all --delta-yc=1 --epsilon-yc=1 --delta-yhat=1 --epsilon-yhat=1 --cat-yyhat")

print("y,c,yhat: delta=1.5, epsilon=0 (kurt: 1.3 skew: 0)")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbb_partial_only-kurt-all --delta-yc=1.5 --epsilon-yc=0 --delta-yhat=1.5 --epsilon-yhat=0 --cat-yyhat")

print("normal y, c; non-normal yhat: delta=2, epsilon=2 (kurt: 10 skew: -3)")
os.system("python3 " + dir + "/simulation.py --out_prefix=bbb_partial_non-normal-yhat --delta-yhat=2 --epsilon-yhat=3 --cat-yyhat")