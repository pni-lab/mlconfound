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
    return fit.rsquared.flatten()[0]


def _r2_cont_cat(x, y):
    return _r2_cat_cont(y, x)


def _r2_cat_cat(x, y):
    df = pd.DataFrame({
        'x': x,
        'y': y
    })
    fit = mnlogit('y ~ C(x)', data=df).fit(disp=0, method='powell')
    return fit.prsquared


def _r2_cat_cat_r(x, y):
    return _r2_cat_cat(y, x)


def _r2_factory(cat_x, cat_y, reverse_cat=False):
    if cat_x and cat_y:
        if reverse_cat:
            return _r2_cat_cat_r
        else:
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


def _mnlogit_cdf(fit, df):
    mat = np.log(fit.predict(df))  # predict returns class probabilities
    mat.index = df.Z
    labels = np.unique(df.X)
    columns = [np.argwhere(labels == i).flatten()[0] for i in df.X]  # label 2 index
    return mat.iloc[:, columns].values.T


def _gauss_cdf(fit, df):
    mu = np.array(fit.predict(df))
    resid = df.X.values - mu
    sigma = np.repeat(np.std(resid), len(df.Z.values))
    # X | Z = Z_i ~ N(mu[i], sig2[i])
    return np.array([norm.logpdf(df.X.values, loc=m, scale=sigma) for m in mu]).T


def _conditional_log_likelihood_gaussian_cont_cont(X0, Z):
    df = pd.DataFrame({
        'Z': Z,
        'X': X0
    })
    fit = ols('X ~ Z', data=df).fit()
    return _gauss_cdf(fit, df)


def _conditional_log_likelihood_gaussian_cont_cat(X0, Z):
    df = pd.DataFrame({
        'Z': Z,
        'X': X0
    })
    fit = ols('X ~ C(Z)', data=df).fit()
    return _gauss_cdf(fit, df)


def _conditional_log_likelihood_gaussian_cat_cont(X0, Z):
    df = pd.DataFrame({
        'Z': Z,
        'X': X0
    })
    fit = mnlogit('X ~ Z', data=df).fit(disp=0, method='powell')
    return _mnlogit_cdf(fit, df)


def _conditional_log_likelihood_gaussian_cat_cat(X0, Z):
    df = pd.DataFrame({
        'Z': Z,
        'X': X0
    })
    fit = mnlogit('X ~ C(Z)', data=df).fit(disp=0, method='powell')
    return _mnlogit_cdf(fit, df)


def _conditional_log_likelihood_factory(cat_x, cat_y):
    if cat_x and cat_y:
        return _conditional_log_likelihood_gaussian_cat_cat
    elif cat_x:
        return _conditional_log_likelihood_gaussian_cat_cont
    elif cat_y:
        return _conditional_log_likelihood_gaussian_cont_cat
    else:
        return _conditional_log_likelihood_gaussian_cont_cont


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


def cpt(x, y, z, t_xy, t_xz, t_yz, condlike_f, num_perms=1000, mcmc_steps=50, cond_dist_method='GaussianReg',
        return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    if cond_dist_method != 'GaussianReg':
        assert NotImplementedError("Currently only regression-based Gaussian conditional distribution estimation "
                                   "('GaussianReg') is implemented.")

    rng = np.random.default_rng(random_state)
    random_sates = rng.integers(np.iinfo(np.int32).max, size=num_perms)

    x = np.array(x)

    r2_x_z = t_xz(x, z)
    r2_y_z = t_yz(y, z)
    r2_x_y = t_xy(x, y)

    cond_log_lik_mat = condlike_f(x, z)
    Pi_init = _generate_X_CPT_MC(mcmc_steps, cond_log_lik_mat, np.arange(len(x), dtype=int), random_state=random_state)

    def workhorse(_random_state):
        # batched os job_batch for efficient parallelization
        Pi = _generate_X_CPT_MC(mcmc_steps, cond_log_lik_mat, Pi_init, random_state=_random_state)
        return t_xy(x[Pi], y)

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


def full_confound_test(y, yhat, c, num_perms=1000, cat_y=False, cat_yhat=False, cat_c=False, mcmc_steps=50,
                       return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    """
       The Full Confounder Test, to test for full confounder bias in machine learning models,
       based on the target variable, the confounder and the model predictions. Low p-value indicates that the model
       predictions are not fully driven by the confoudner.

       Notes
       -----
       Performs partial confounder test, a statistical test described in [1]_ and
       based on the conditional permutation test for independence [2]_.
       The null hypothesis of the test is that the model predictions are independent of the target variable,
       given the confounder variable, i.e. the model is entirely driven by the confounder.
       A low p-value therefore indicates that the model is not fully driven by the confounder.

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
       mcmc_steps : int
           Number of steps for the Markov-chain Monte Carlo sampling.
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
       ResultsFullyConfounded
           Named tuple with the fields:

           - "r2_y_c": coefficient-of-determination between y and c,

           - "r2_y_yhat": coefficient-of-determination between y and yhat,

           - "r2_yhat_c": coefficient-of-determination between yhat and c,

           - "p": p-value,

           - "p_ci": binomial 95% confidence interval for the p-value,

           - "null_distribution": numpy.ndarray containing the permuted null distribution or None depending on teh value of
                   return_null_dist.

       Examples
       --------
       See `notebooks/quickstart.ipynb` for more detailed examples.

       >>> partial_confound_test(y=[1,2,3,4,5,6], yhat=[1.5,2.3,2.9,4.2,5,5.7], c=[3,5,4,6,1,2], random_state=42).p
       0.693

       See Also
       --------
       test_partially_confounded

       References
       ----------
       [1] Spisak, Tamas "A conditional permutation-based approach to test confounder effect
        and center-bias in machine learning models", in prep.

       [2] Berrett, Thomas B., et al. "The conditional permutation test for independence
       while controlling for confounders."
       Journal of the Royal Statistical Society: Series B (Statistical Methodology) 82.1 (2020): 175-197.

    """

    r2_y_yhat = _r2_factory(cat_y, cat_yhat, reverse_cat=True)
    r2_y_c = _r2_factory(cat_y, cat_c, reverse_cat=True)
    r2_yhat_c = _r2_factory(cat_yhat, cat_c, reverse_cat=True)

    condlike_f = _conditional_log_likelihood_factory(cat_y, cat_yhat)

    return ResultsFullyConfounded(
        *cpt(x=y, y=yhat, z=c, num_perms=num_perms, t_xy=r2_y_yhat, t_xz=r2_y_c, t_yz=r2_yhat_c, condlike_f=condlike_f,
             mcmc_steps=mcmc_steps, return_null_dist=return_null_dist, random_state=random_state,
             progress=progress, n_jobs=n_jobs))


ResultsPartiallyConfounded = namedtuple('ResultsPartiallyConfounded', ['r2_y_c',
                                                                       'r2_yhat_c',
                                                                       'r2_y_yhat',
                                                                       'p',
                                                                       'p_ci',
                                                                       'null_distribution'])


def partial_confound_test(y, yhat, c, num_perms=1000, cat_y=False, cat_yhat=False, cat_c=False, mcmc_steps=50,
                          return_null_dist=False, random_state=None, progress=True, n_jobs=-1):
    """
    The Partial Confounder Test, to test partial confounder bias in machine learning models,
    based on the target variable, the confounder and the model predictions. Low p-value indicates that the model
    predictions are partially driven by the confoudner.

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
    mcmc_steps : int
        Number of steps for the Markov-chain Monte Carlo sampling.
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

        - "r2_y_c": coefficient-of-determination between y and c,

        - "r2_yhat_c": coefficient-of-determination between yhat and c,

        - "r2_y_yhat": coefficient-of-determination between y and yhat,

        - "p": p-value,

        - "p_ci": binomial 95% confidence interval for the p-value,

        - "null_distribution": numpy.ndarray containing the permuted null distribution or None depending on teh value of
                return_null_dist.

    Examples
    --------
    See `notebooks/quickstart.ipynb` for more detailed examples.

    >>> partial_confound_test(y=[1,2,3,4,5,6], yhat=[1.5,2.3,2.9,4.2,5,5.7], c=[3,5,4,6,1,2], random_state=42).p
    0.693

    See Also
    --------
    test_fully_confounded

    References
    ----------
    [1] Spisak, Tamas "A conditional permutation-based approach to test confounder effect
     and center-bias in machine learning models", in prep.

    [2] Berrett, Thomas B., et al. "The conditional permutation test for independence
    while controlling for confounders."
    Journal of the Royal Statistical Society: Series B (Statistical Methodology) 82.1 (2020): 175-197.

    """

    r2_c_yhat = _r2_factory(cat_c, cat_yhat)
    r2_c_y = _r2_factory(cat_c, cat_y)
    r2_yhat_y = _r2_factory(cat_yhat, cat_y)

    condlike_f = _conditional_log_likelihood_factory(cat_c, cat_yhat)

    return ResultsPartiallyConfounded(
        *cpt(x=c, y=yhat, z=y, num_perms=num_perms, t_xy=r2_c_yhat, t_xz=r2_c_y, t_yz=r2_yhat_y, condlike_f=condlike_f,
             mcmc_steps=mcmc_steps, return_null_dist=return_null_dist, random_state=random_state,
             progress=progress, n_jobs=n_jobs))
