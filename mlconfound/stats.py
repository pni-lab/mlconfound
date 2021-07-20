from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.stats import beta
from statsmodels.formula.api import ols, logit

from ._utils import tqdm_joblib


def _r2_cont_cont(x, y):
    # faster than scipy or statmodels
    return np.corrcoef(x, y)[0, 1] ** 2


def _r2_cat_cont(x, y):
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    fit = ols('y ~ C(x)', data=df).fit()
    return fit.rsquared


def _r2_cont_cat(x, y):
    return _r2_cat_cont(y, x)


def _r2_cat_cat(x, y):
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    fit = logit('C(y) ~ C(x)', data=df).fit()
    return fit.rsquared


def _binom_ci(success, total, ci=0.95):
    quantile = (1 - ci) / 2
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 0
    return lower, upper


def conditional_permutation_gaussian(X0, Z, X_cat=False, Z_cat=False, nstep=10, M=1000, random_state=None):
    df = pd.DataFrame({
        'Z': Z,
        'X': X0
    })
    if X_cat and Z_cat:
        fit = logit('C(X) ~ C(Z)', data=df).fit()
    elif X_cat:
        fit = logit('C(X) ~ Z', data=df).fit()
    elif Z_cat:
        fit = ols('X ~ C(Z)', data=df).fit()
    else:
        fit = ols('X ~ Z', data=df).fit()
    mu = fit.predict(df)
    resid = X0 - mu
    sigma2 = np.repeat(np.power(np.std(resid), 2), len(Z))
    # X | Z = Z_i ~ N(mu[i], sig2[i])
    log_lik_mat = - np.power(X0, 2)[:, None] * (1 / 2 / sigma2)[None, :] + X0[:, None] * (mu / sigma2)[None, :]
    Pi_mat = _generate_X_CPT(nstep, M, log_lik_mat, random_state=random_state)
    return X0[Pi_mat]


def _generate_X_CPT(nstep, M, log_lik_mat, Pi_init=[], random_state=None):
    # modified version of: http: // www.stat.uchicago.edu / ~rina / cpt / Bikeshare1.html
    # Berrett, T.B., Wang, Y., Barber, R.F. and Samworth, R.J., 2020. The conditional permutation test
    # for independence while controlling for confounders.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1), pp.175 - 197.
    n = log_lik_mat.shape[0]
    if len(Pi_init) == 0:
        Pi_init = np.arange(n, dtype=int)
    Pi_ = _generate_X_CPT_MC(nstep, log_lik_mat, Pi_init, random_state=random_state)
    Pi_mat = np.zeros((M, n), dtype=int)
    for m in range(M):
        Pi_mat[m] = _generate_X_CPT_MC(nstep, log_lik_mat, Pi_, random_state=random_state)
    return Pi_mat


def _generate_X_CPT_MC(nstep, log_lik_mat, Pi, random_state=None):
    # modified version of: http: // www.stat.uchicago.edu / ~rina / cpt / Bikeshare1.html
    # Berrett, T.B., Wang, Y., Barber, R.F. and Samworth, R.J., 2020. The conditional permutation test
    # for independence while controlling for confounders.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 82(1), pp.175 - 197.
    n = len(Pi)
    npair = np.floor(n/2).astype(int)
    rng = np.random.default_rng(random_state)
    for istep in range(nstep):
        perm = rng.choice(n, n, replace=False)
        inds_i = perm[0:npair]
        inds_j = perm[npair:(2*npair)]
        # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
        log_odds = log_lik_mat[Pi[inds_i], inds_j] + log_lik_mat[Pi[inds_j], inds_i] \
            - log_lik_mat[Pi[inds_i], inds_i] - log_lik_mat[Pi[inds_j], inds_j]
        swaps = rng.binomial(1, 1/(1+np.exp(-np.maximum(-500, log_odds))))
        Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps*(Pi[inds_j]-Pi[inds_i]), Pi[inds_j] - \
            swaps*(Pi[inds_j]-Pi[inds_i])
    return Pi


ConfoundTestResults = namedtuple('ConfoundTestResults', ['r2_y_c',
                                                         'r2_yhat_c',
                                                         'r2_y_yhat',
                                                         'p',
                                                         'p_ci',
                                                         'found'])


ConfoundTestResultsDetailed = namedtuple('ConfoundTestResults', ['r2_y_c',
                                                                 'r2_yhat_c',
                                                                 'r2_y_yhat',
                                                                 'p',
                                                                 'p_ci',
                                                                 'found',
                                                                 'null_distribution'])


def confound_test(y, yhat, c,
                  num_perms=10000,
                  cat_y=False,
                  cat_c=False,
                  nstep=10,
                  return_null_dist=False,
                  random_state=None,
                  progress=True,
                  n_jobs=-1):
    if cat_y and cat_c:
        r2_yc = _r2_cat_cat
        r2_yy = _r2_cat_cat
    elif cat_y:
        r2_yc = _r2_cat_cont
        r2_yy = _r2_cat_cat
    elif cat_c:
        r2_yc = _r2_cont_cat
        r2_yy = _r2_cont_cont
    else:
        r2_yc = _r2_cont_cont
        r2_yy = _r2_cont_cont

    rng = np.random.default_rng(random_state)
    random_sates = rng.integers(np.iinfo(np.int32).max, size=num_perms)

    r2_y_c = r2_yc(y, c)
    r2_y_yhat = r2_yy(y, yhat)
    r2_yhat_c = r2_yc(yhat, c)

    def workhorse(_random_state):
        # batched os job_batch for efficient parallelization
        c_star = conditional_permutation_gaussian(c, y, nstep=nstep, M=1, random_state=None)
        return r2_yc(yhat, c_star.flatten())

    with tqdm_joblib(tqdm(desc='Permuting', total=num_perms, disable=not progress)):
        r2_yhat_c_star = np.array(Parallel(n_jobs=n_jobs)(delayed(workhorse)(i) for i in random_sates))

    p = np.sum(r2_yhat_c_star >= r2_yhat_c) / len(r2_yhat_c_star)
    ci = _binom_ci(len(r2_yhat_c_star) * p, len(r2_yhat_c_star))

    if return_null_dist:
        return ConfoundTestResultsDetailed(
            r2_y_c,
            r2_yhat_c,
            r2_y_yhat,
            p,
            ci,
            len(r2_yhat_c_star),
            r2_yhat_c_star
        )
    else:
        return ConfoundTestResults(
            r2_y_c,
            r2_yhat_c,
            r2_y_yhat,
            p,
            ci,
            len(r2_yhat_c_star)
        )
