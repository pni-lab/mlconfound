#!/usr/bin/env python
import argparse

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

from mlconfound.simulate import simulate_y_c_X
from mlconfound.stats import confound_test

parser = argparse.ArgumentParser(description="Validate 'mlconfound' on simulated data.")

parser.add_argument("-r", "--repeat", help="Repeat r times. (seed incremented, if not None)",
                    action="store", type=int, default=1)
parser.add_argument("--header", help="Print header and exit.",
                    action="store_true")
parser.add_argument("-j", "--n_jobs", dest="n_jobs", action="store", default=-1,
                    help="Number of cores to use. Default: -1", type=int)

arg_simulate = parser.add_argument_group('General options for the simulation')
arg_simulate.add_argument('-n', '--n', '--number-of-observations', dest="n", action="store",
                          default=100,
                          help="Number of observations (default: 100)", type=int)
arg_simulate.add_argument("-s", "--seed", "--random-seed", dest="random_seed", action="store", default=None,
                          help="Seed for random generator. (default: None)", type=int)

arg_simulate_target = parser.add_argument_group('Options for target (y) simulation')
arg_simulate_target.add_argument("-y", "--target_type", dest="target_type", action="store", default=0,
                                 help="Levels of target variable. 0 means continuous (default: 0)", type=int)
arg_simulate_target.add_argument("--ts_ratio_y", dest="ts_ratio_y", action="store", default=0.6,
                                 help="Ratio of true signal in in y (default: 0.6)", type=float)
arg_simulate_target.add_argument("--cs_ratio_y", dest="cs_ratio_y", action="store", default=0.0,
                                 help="Ratio of confounder signal in in y (default: 0.0)", type=float)

arg_simulate_confounder = parser.add_argument_group('Options for confounder (c) simulation')
arg_simulate_confounder.add_argument("-c", "--confound_type", dest="confound_type", action="store", default=0,
                                     help="Levels of confounder variable. 0 means continuous (default: 0)", type=int)
arg_simulate_confounder.add_argument("--ts_ratio_c", dest="ts_ratio_c", action="store", default=0.2,
                                     help="Ratio of true signal in in c (default: 0.0)", type=float)
arg_simulate_confounder.add_argument("--cs_ratio_c", dest="cs_ratio_c", action="store", default=0.7,
                                     help="Ratio of confounder signal in in c (default: 0.8)", type=float)

arg_simulate_feature = parser.add_argument_group('Options for feature (X) simulation')
arg_simulate_feature.add_argument("-p", "--p", "--number-of-features", dest="n_features", action="store",
                                  default=20,
                                  help="Number of features (default: 20)", type=int)
arg_simulate_feature.add_argument("--ts_ratio_x", dest="ts_ratio_X", action="store", default=0.6,
                                  help="Ratio of true signal in in X (default: 0.6)", type=float)
arg_simulate_feature.add_argument("--cs_ratio_x", dest="cs_ratio_X", action="store", default=0.0,
                                  help="Ratio of confounder signal in in X (default: 0.0)", type=float)
arg_simulate_feature.add_argument("--sparsity", dest="dirichlet_sparsity", action="store", default=1.0,
                                  help="Sparsity of the feature matrix, "
                                       "as determined by the Dirichlet-alpha (default: 1.0)",
                                  type=float)
arg_simulate_feature.add_argument("--corr", dest="X_corr", action="store", default=0.0,
                                  help="Correlation of the noise in the feature matrix (default: 0.0)", type=float)

arg_ml = parser.add_argument_group('Options for the ML-model')
arg_ml.add_argument("--models", dest="models", action="store",
                    default='RidgeCV,LassoCV,RandomForestRegressor',
                    help="Models to train. Possible values: RidgeCV,LassoCV,RandomForestRegressor. "
                         "Defauls: 'RidgeCV,LassoCV,RandomForestRegressor'",
                    type=str)

arg_confound_test = parser.add_argument_group('Options confound testing')
arg_confound_test.add_argument("--num_perms", dest="num_perms", action="store",
                               default=1000,
                               help="Number of permutations. Default: 1000",
                               type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.header:
        print("p, r2_y_c, r2_y_c, r2_y_yhat,"
              "random_seed, n,"
              "ts_ratio_y, cs_ratio_y,"
              "ts_ratio_c, cs_ratio_c,"
              "ts_ratio_X, cs_ratio_X,"
              "n_features,"
              "X_corr,"
              "dirichlet_sparsity,"
              "model,"
              "num_perms"
              )
        exit(0)

    for r in range(args.repeat):

        if args.random_seed is not None:
            args.random_seed += 1

        # simulation
        y, c, X = simulate_y_c_X(args.ts_ratio_y, args.cs_ratio_y,
                                 args.ts_ratio_c, args.cs_ratio_c,
                                 args.ts_ratio_X, args.cs_ratio_X,
                                 args.n_features, args.X_corr,
                                 args.dirichlet_sparsity,
                                 n=args.n, random_state=args.random_seed)

        # do nested cross validations
        yhat = {}
        if 'RidgeCV' in args.models:
            yhat['RidgeCV'] = cross_val_predict(RidgeCV(), X, y, n_jobs=args.n_jobs)
        if 'LassoCV' in args.models:
            yhat['LassoCV'] = cross_val_predict(LassoCV(), X, y, n_jobs=args.n_jobs)
        if 'RandomForestRegressor' in args.models:
            yhat['RandomForestRegressor'] = cross_val_predict(RandomForestRegressor(random_state=args.random_seed),
                                                              X, y, n_jobs=args.n_jobs)

        # do confounder testing
        for model in yhat:
            test_res = confound_test(y, yhat[model], c, num_perms=args.num_perms,
                                     progress=False, n_jobs=args.n_jobs, random_state=args.random_seed)

            res = [test_res.p, test_res.r2_y_c, test_res.r2_y_c, test_res.r2_y_yhat,
                   args.random_seed, args.n,
                   args.ts_ratio_y, args.cs_ratio_y,
                   args.ts_ratio_c, args.cs_ratio_c,
                   args.ts_ratio_X, args.cs_ratio_X,
                   args.n_features,
                   args.X_corr,
                   args.dirichlet_sparsity,
                   model,
                   args.num_perms]
            print(','.join([str(element) for element in res])
)

