from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import beta, norm
from statsmodels.formula.api import ols, mnlogit
from tqdm import tqdm

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
    fit = mnlogit('y ~ C(x)', data=df).fit(disp=0)
    return fit.prsquared


def _r2_factory(cat_x, cat_y):
    if cat_x and cat_y:
        return _r2_cat_cat
    elif cat_x:
        return _r2_cat_cont
    elif cat_y:
        return _r2_cont_cat
    else:
        return _r2_cont_cont


def _binom_ci(success, total, ci=0.95):
    quantile = (1 - ci) / 2
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 0
    return lower, upper


def _conditional_log_likelihood_gaussian(X0, Z, X_cat=False, Z_cat=False):
    df = pd.DataFrame({
        'Z': Z,
        'X': X0
    })

    if X_cat:
        if Z_cat:
            fit = mnlogit('X ~ C(Z)', data=df).fit(disp=0)
        else:
            fit = mnlogit('X ~ Z', data=df).fit(disp=0)
        mat = np.log(fit.predict(df))  # predict returns class probabilities
        mat.index = df.Z
        labels = np.unique(df.X)
        columns = [np.argwhere(labels == i).flatten()[0] for i in df.X]  # label 2 index
        return mat.iloc[:, columns].values.T
    else:
        if Z_cat:
            fit = ols('X ~ C(Z)', data=df).fit()
        else:
            fit = ols('X ~ Z', data=df).fit()
        mu = np.array(fit.predict(df))
        resid = X0 - mu
        sigma = np.repeat(np.std(resid), len(Z))
        # X | Z = Z_i ~ N(mu[i], sig2[i])
        return np.array([norm.logpdf(X0, loc=m, scale=sigma) for m in mu]).T
    # return -np.power(X0, 2)[:, None] * (1 / 2 / sigma2)[None, :] + X0[:, None] * (mu / sigma2)[None, :]


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
    npair = np.floor(n / 2).astype(int)
    rng = np.random.default_rng(random_state)
    for istep in range(nstep):
        perm = rng.choice(n, n, replace=False)
        inds_i = perm[0:npair]
        inds_j = perm[npair:(2 * npair)]
        # for each k=1,...,npair, decide whether to swap Pi[inds_i[k]] with Pi[inds_j[k]]
        log_odds = log_lik_mat[Pi[inds_i], inds_j] + log_lik_mat[Pi[inds_j], inds_i] \
                   - log_lik_mat[Pi[inds_i], inds_i] - log_lik_mat[Pi[inds_j], inds_j]
        swaps = rng.binomial(1, 1 / (1 + np.exp(-np.maximum(-500, log_odds))))
        Pi[inds_i], Pi[inds_j] = Pi[inds_i] + swaps * (Pi[inds_j] - Pi[inds_i]), Pi[inds_j] - \
                                 swaps * (Pi[inds_j] - Pi[inds_i])
    return Pi


CptResults = namedtuple('CptResults', ['r2_x_z',
                                       'r2_x_y',
                                       'r2_y_z',
                                       'p',
                                       'p_ci',
                                       'null_distribution'])


def cpt(x, y, z, num_perms=1000, cat_x=False, cat_y=False, cat_z=False, mcmc_nstep=50, cond_dist_method='GaussianReg',
        return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    if cond_dist_method != 'GaussianReg':
        assert NotImplementedError("Currently only regression-based Gaussian conditional distribution estimation "
                                   "('GaussianReg') is implemented.")

    r2_xy = _r2_factory(cat_x, cat_y)
    r2_xz = _r2_factory(cat_x, cat_z)
    r2_yz = _r2_factory(cat_y, cat_z)

    rng = np.random.default_rng(random_state)
    random_sates = rng.integers(np.iinfo(np.int32).max, size=num_perms)

    x = np.array(x)

    r2_x_z = r2_xz(x, z)
    r2_y_z = r2_yz(y, z)
    r2_x_y = r2_xy(x, y)

    cond_log_lik_mat = _conditional_log_likelihood_gaussian(x, z, X_cat=cat_x, Z_cat=cat_z)
    Pi_init = _generate_X_CPT_MC(mcmc_nstep, cond_log_lik_mat, np.arange(len(x), dtype=int), random_state=random_state)

    def workhorse(_random_state):
        # batched os job_batch for efficient parallelization
        Pi = _generate_X_CPT_MC(mcmc_nstep, cond_log_lik_mat, Pi_init, random_state=_random_state)
        return r2_xy(x[Pi], y)

    with tqdm_joblib(tqdm(desc='Permuting', total=num_perms, disable=not progress)):
        r2_xpi_y = np.array(Parallel(n_jobs=n_jobs)(delayed(workhorse)(i) for i in random_sates))

    p = np.sum(r2_xpi_y >= r2_x_y) / len(r2_xpi_y)
    ci = _binom_ci(len(r2_xpi_y) * p, len(r2_xpi_y))

    if not return_null_dist:
        r2_xpi_y = None

    return CptResults(
        r2_x_z,
        r2_x_y,
        r2_y_z,
        p,
        ci,
        r2_xpi_y
    )


ResultsFullyConfounded = namedtuple('ResultsFullyConfounded', ['r2_y_c',
                                                               'r2_y_yhat',
                                                               'r2_yhat_c',
                                                               'p',
                                                               'p_ci',
                                                               'null_distribution'])


def test_fully_confounded(y, yhat, c, num_perms=1000, cat_y=False, cat_yhat=False, cat_c=False, mcmc_nstep=50,
                          cond_dist_method='GaussianReg',
                          return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    return ResultsFullyConfounded(
        *cpt(x=y, y=yhat, z=c, num_perms=num_perms, cat_x=cat_y, cat_y=cat_yhat, cat_z=cat_c, mcmc_nstep=mcmc_nstep,
             cond_dist_method=cond_dist_method, return_null_dist=return_null_dist, random_state=random_state,
             progress=progress, n_jobs=n_jobs))


ResultsPartiallyConfounded = namedtuple('ResultsPartiallyConfounded', ['r2_y_c',
                                                                       'r2_yhat_c',
                                                                       'r2_y_yhat',
                                                                       'p',
                                                                       'p_ci',
                                                                       'null_distribution'])


def test_partially_confounded(y, yhat, c, num_perms=1000, cat_y=False, cat_yhat=False, cat_c=False, mcmc_nstep=50,
                              cond_dist_method='GaussianReg',
                              return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    """
    Partial Confounder Test

    Notes
    -----
    Performs partial confounder test, a statistical test described in [1]_ and
    based on the conditional permutation test for independence [2]_.
    The null hypothesis of the test is that the model predictions are independent of the confounder,
    given the target variable, i.e. there is no-confounder bias.
    A low p-value therefore indicates significant confounder bias.

    The method has no assumptions about the distribution of yhat, but assumes normality for the conditional distribution
     (c | y). It is however, fairly robust to violating this assumptions, see [1]_.


    Parameters
    ----------
    y : array_like
        Target variable.
    yhat : array_like
        Predictions.
    c : array_like
        Confounder variable.
    num_perms : int
        Number of conditional permutations.
    cat_y : bool
        Flag for categorical target variable (classification).
    cat_yhat : bool
        Flag for categorical predictions (classification). Must be false, if class probabilities are used.
    cat_c : bool
        Flag for categorical confounder variable (e.g center, sex or batch).
    mcmc_nstep : int
        Number of steps for the Markov-chain Monte Carlo sampling.
    cond_dist_method {GaussianReg}
        Method to estimate the conditional distribution. Currently only Gaussian estimation is supported.
    return_null_dist : bool
        Whether permuted null distribution should be returned (e.g. for plotting purposes).
    random_state : int or None
        Random state for the conditional permutation.
    progress : bool
        Whether to print out progress bar.
    n_jobs : int
        Number of parallel jobs (-1: use all available processors).

    Returns
    -------
    ResultsPartiallyConfounded
        Named tuple with the fields:
        - r2_y_c: coefficient-of-determination between y and c
        - r2_yhat_c: coefficient-of-determination between yhat and c
        - r2_y_yhat: coefficient-of-determination between y and yhat
        - p: p-value
        - p_ci: binomial 95% confidence interval for the p-value
        - null_distribution: numpy.ndarray containing the permuted null distribution or None depending on teh value of
          return_null_dist

    References
    ----------
    [2] Spisak, Tamas "A conditional permutation-based approach to test confounder effect
     and center-bias in machine learning models", in prep.

    [2] Berrett, Thomas B., et al. "The conditional permutation test for independence
    while controlling for confounders."
    Journal of the Royal Statistical Society: Series B (Statistical Methodology) 82.1 (2020): 175-197.

    """
    return ResultsPartiallyConfounded(
        *cpt(x=c, y=yhat, z=y, num_perms=num_perms, cat_x=cat_c, cat_y=cat_yhat, cat_z=cat_y, mcmc_nstep=mcmc_nstep,
             cond_dist_method=cond_dist_method, return_null_dist=return_null_dist, random_state=random_state,
             progress=progress, n_jobs=n_jobs))
