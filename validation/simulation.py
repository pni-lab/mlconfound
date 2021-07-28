#!/usr/bin/env python
import argparse

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

import os
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys.path[0] + '/..')
from mlconfound.stats import test_fully_confounded, test_partially_confounded
from mlconfound.simulate import simulate_y_c_yhat, sinh_arcsinh

parser = argparse.ArgumentParser(description="Validate 'mlconfound' on simulated data.")

parser.add_argument("--mode", help="Confound testing mode: 'partial' or 'full'. Default: 'partial'. ",
                    choices=['partial', 'full'], type=str, default='partial')

parser.add_argument("--delta-yc", dest="delta_yc", action="store", default=1.0,
                    help="Delta of the sinh_arcsinh transformation on y and c to set kurtosis. "
                         "Default: 1: no transformation",
                    type=float)
parser.add_argument("--epsilon-yc", dest="epsilon_yc", action="store", default=0.0,
                    help="Epsilon of the sinh_arcsinh transformation on y and c to set skewness. "
                         "Default: 0: no transformation",
                    type=float)

parser.add_argument("--delta-yhat", dest="delta_yhat", action="store", default=1.0,
                    help="Delta of the sinh_arcsinh transformation on yhat to set kurtosis. "
                         "Default: 1: no transformation",
                    type=float)
parser.add_argument("--epsilon-yhat", dest="epsilon_yhat", action="store", default=0.0,
                    help="Epsilon of the sinh_arcsinh transformation on yhat to set skewness. "
                         "Default: 0: no transformation",
                    type=float)

parser.add_argument("--cat-yyhat", help="y and yhat is categorical. (default: continuous)",
                    action="store_true")
parser.add_argument("--cat-c", help="c is categorical. (default: continuous)",
                    action="store_true")

parser.add_argument("--random-seed", dest="random_seed", action="store", default=4242,
                    help="Random seed. Default: 4242", type=int)
parser.add_argument("--n-jobs", dest="n_jobs", action="store", default=-1,
                    help="Number of cores to use. Default: -1", type=int)

parser.add_argument("--out_prefix", dest="out_prefix", action="store", default='validation',
                    help="Output file name prefix.", type=str)

parser.add_argument("--out", dest="out_file", action="store", default=None,
                    help="Output file name. Overrides --out_prefix", type=str)



if __name__ == '__main__':
    args = parser.parse_args()

    if args.mode == 'partial':  # option 3: partial
        ################# default simulation parameters #########################
        confound_test = test_partially_confounded

        repetitions = 100
        num_perms = 1000

        all_n = [50, 100, 500, 1000]

        all_cov_y_c = [0, 0.2, 0.4, 0.6, 0.8]

        all_c_to_y_ratio_in_yhat = [0, 0.1, 0.3, 0.5, 1]
        all_yc_ratio = all_c_to_y_ratio_in_yhat

        all_yc_in_yhat = [0, 0.3, 0.6, 0.9]

        #########################################################################
    elif args.mode == 'full':  #option 1: full
        ################# default simulation parameters #########################
        confound_test = test_fully_confounded

        repetitions = 100
        num_perms = 1000

        all_n = [50, 100, 500, 1000]

        all_cov_y_c = [0, 0.2, 0.4, 0.6, 0.8]

        all_y_to_c_ratio_in_yhat = [0, 0.1, 0.3, 0.5, 1]
        all_yc_ratio = all_y_to_c_ratio_in_yhat

        all_yc_in_yhat = [0, 0.3, 0.6, 0.9]

        #########################################################################
    else:
        raise AttributeError('No such mode', args.mode)

    print('Number of simulations:', np.prod([len(i) for i in [
        all_cov_y_c,
        all_yc_in_yhat,
        all_yc_ratio,
        all_n]]) * repetitions)

    all_param_configurations = itertools.product(
        all_cov_y_c,
        all_yc_in_yhat,
        all_yc_ratio,
        all_n)

    rng = np.random.default_rng(args.random_seed)

    results = pd.DataFrame(columns=["p", "r2_y_c", "r2_yhat_c", "r2_y_yhat",
                                    "n", "c_to_y_ratio_in_yhat", "yc_in_yhat", "cov_y_c",
                                    "num_perms", "random_seed"])

    for cov_y_c, yc_in_yhat, yc_ratio, n in tqdm(list(all_param_configurations)):

        if args.mode == 'partial':
            y_ratio_yhat = np.round(yc_in_yhat / (yc_ratio + 1), 2)
            c_ratio_yhat = np.round(yc_ratio * yc_in_yhat / (yc_ratio + 1), 2)
        elif args.mode == 'full':
            c_ratio_yhat = np.round(yc_in_yhat / (yc_ratio + 1), 2)
            y_ratio_yhat = np.round(yc_ratio * yc_in_yhat / (yc_ratio + 1), 2)

        def workhorse(_random_state):
            # simulate
            y, c, yhat = simulate_y_c_yhat(
                cov_y_c=cov_y_c,
                y_ratio_yhat=y_ratio_yhat,
                c_ratio_yhat=c_ratio_yhat,
                n=n,
                random_state=_random_state)

            # introduce non-normality
            y = sinh_arcsinh(y, args.delta_yc, args.epsilon_yc)
            c = sinh_arcsinh(c, args.delta_yc, args.epsilon_yc)
            yhat = sinh_arcsinh(yhat, args.delta_yhat, args.epsilon_yhat)

            # is it categorical?
            # right now only binary variables can be simulated
            if args.cat_yyhat:
                y = y > 0
                yhat = yhat > 0
            if args.cat_c:
                c = c > 0

            # test
            res = confound_test(y, yhat, c,
                                cat_y=args.cat_yyhat,
                                cat_yhat=args.cat_yyhat,
                                cat_c=args.cat_c,
                                num_perms=num_perms,
                                random_state=_random_state,
                                n_jobs=1,
                                progress=False)
            # return
            return res.p, res.r2_y_c, res.r2_yhat_c, res.r2_y_yhat, _random_state


        random_sates = rng.integers(np.iinfo(np.int32).max, size=repetitions)
        p, r2_y_c, r2_yhat_c, r2_y_yhat, random_seed = zip(*
                                                           Parallel(n_jobs=-1)(
                                                               delayed(workhorse)(rs) for rs in random_sates)
                                                           )

        if args.mode == 'partial':
            name = "c_to_y_ratio_in_yhat"
        elif args.mode == 'full':
            name = "y_to_c_ratio_in_yhat"
        # create DataFrame and save it
        results = results.append(pd.DataFrame({
            "p": p,
            "r2_y_c": r2_y_c,
            "r2_yhat_c": r2_yhat_c,
            "r2_y_yhat": r2_y_yhat,
            "n": n,
            name: yc_ratio,
            "yc_in_yhat": yc_in_yhat,
            "cov_y_c": cov_y_c,
            "num_perms": num_perms,
            "random_seed": list(random_seed)
        }), ignore_index=True)

        # update file after every iteration...
        if args.out_file is None:
            params = '_'.join(str(args)[10:].split(', '))[:-1]
            filename = args.out_prefix + '_' + params + '.csv'
            results.to_csv(sys.path[0] + '/data_out/' + filename)
        else:
            results.to_csv(args.out_file)

