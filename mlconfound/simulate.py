import numpy as np
from warnings import warn


def sinh_arcsinh(x, delta=1, epsilon=0):
    """
    Sinh-arcsinh transformation

    Notes
    -----
    The sinh-arcsinh transformation of Jones and Pewsey [1]_ can be used to transfrom Normal distribution to non-normal.

    Parameters
    ----------
    x : array_like
        Normally distributed input data.
    delta : float
        Parameter to control kurtosis, delta=1 means no change.
    epsilon : float
        Parameter to control skewness, epsilon=0 means no change.
    Returns
    -------
    array_like
        Transformed data.

    Examples
    --------
    See `validation/simulation.py` for an application example.

    >>> result = sinh_arcsinh([-1, -0.5, -0.1, 0.1, 0.5, 1], delta=2, epsilon=1)
    >>> print(result)
    [-7.8900947  -3.48801839 -1.50886059 -0.88854985 -0.03758519  0.83888754]

    >>> result = sinh_arcsinh([-1, -0.5, -0.1, 0.1, 0.5, 1], delta=1, epsilon=0)
    >>> print(result)
    [-1.  -0.5 -0.1  0.1  0.5  1. ]


    See Also
    --------
    simulate_y_c_yhat

    References
    ----------
    [1] Jones, M. C. and Pewsey A. (2009). Sinh-arcsinh distributions. Biometrika 96: 761–780
    """
    return np.sinh(delta * np.arcsinh(x) - epsilon)


def identity(x):
    """
    Identity transformation

    Notes
    -----
    To be used as a transformation function in simulate_y_c_yhat.

    Parameters
    ----------
    x

    Returns
    -------

    See Also
    --------
    simulate_y_c_yhat

    """
    return x


def polynomial(x, coefs=[1, 1]):
    ret = 0
    for c_i, coef in enumerate(coefs):
        ret += coef * np.power(x, c_i + 1)
    return ret


def sigmoid(x, method='tanh'):
    if method == 'tanh':
        return np.tanh(x)
    else:
        raise NotImplementedError("Currently only tanh is implemented.")


def simulate_y_c_yhat(y_ratio_c,
                      y_ratio_yhat, c_ratio_yhat,
                      n,
                      y_delta=1,
                      y_epsilon=0,
                      c_delta=1,
                      c_epsilon=0,
                      yhat_delta=1,
                      yhat_epsilon=0,
                      nonlin_trf_fun=identity,
                      random_state=None):
    """
    Simulate normally distributed target (y), confounder (c) and predictions (yhat).

    Parameters
    ----------
    y_ratio_c: float
        The weight of y in c.
    y_ratio_yhat: float
        The weight of y in yhat.
    c_ratio_yhat: float
        The weight of c in yhat. Set it to zero for H0.
    n: int
        Number of observations.
    y_delta: float
        The delta param of the sinh_archsin transformation on y's contribution in yhat (only affects yhat).
    y_epsilon: float
        The epsilon param of the sinh_archsin transformation on y's contribution in yhat (only affects yhat).
    c_delta: float
        The delta param of the sinh_archsin transformation on c's contribution in yhat (only affects yhat).
    c_epsilon: float
        The epsilon param of the sinh_archsin transformation on c's contribution in yhat (only affects yhat).
    nonlin_trf_fun: callable
        Callable to introduce non-linearity in the conditional distributions. (default: no non-linearity).
    random_state: int
        Numpy random state.

    Returns
    -------
    tuple

        - y: the simulated target variable
        - c: the simulated confounder variable
        - yhat: the simulated predictions

    See Also
    --------
    sinh_arcsinh

    Examples
    --------
    >>> y, c, yhat = simulate_y_c_yhat(0.3, 0.2, 0.2, n=3, random_state=42)
    >>> print(y, c, yhat)
    [ 0.30471708 -1.03998411  0.7504512 ] [ 0.74981043 -1.67771986 -0.6863903 ] [ 0.28760974 -0.73328635  0.00273149]

    """
    rng = np.random.default_rng(random_state)

    y = rng.normal(0, 1, n).T
    c = y_ratio_c * nonlin_trf_fun(y) + rng.normal(0, 1 - y_ratio_c, n)

    # todo: non-normal noise?
    yhat = y_ratio_yhat * nonlin_trf_fun(y) + c_ratio_yhat * nonlin_trf_fun(c) + (
            1 - y_ratio_yhat - c_ratio_yhat) * rng.normal(0, 1, n)

    return sinh_arcsinh(y, delta=y_delta, epsilon=y_epsilon), \
           sinh_arcsinh(c, delta=c_delta, epsilon=c_epsilon), \
           sinh_arcsinh(yhat, delta=yhat_delta, epsilon=yhat_epsilon)


def _create_covariance_matrix(rho, p):
    a = np.repeat(np.arange(p) + 1, p).reshape(p, p)
    b = np.repeat(np.arange(p) + 1, p).reshape(p, p).T
    return rho ** abs(a - b)


def simulate_y_c_X(cov_y_c,
                   y_ratio_X, c_ratio_X,
                   n_features, X_corr,
                   dirichlet_sparsity,
                   n, random_state=None):
    warn('This method is deprecated.', DeprecationWarning, stacklevel=2)

    rng = np.random.default_rng(random_state)

    y, c = rng.multivariate_normal([0, 0], [[1, cov_y_c], [cov_y_c, 1]], n).T

    cov_X = _create_covariance_matrix(X_corr, n_features)

    signs = rng.binomial(1, 0.5, n_features) * 2 - 1

    X = y_ratio_X * y * \
        (rng.dirichlet([dirichlet_sparsity] * n_features, 1) * np.sqrt(n_features) * signs).T
    X += c_ratio_X * c * \
         (rng.dirichlet([dirichlet_sparsity] * n_features, 1) * np.sqrt(n_features) * signs).T
    X += (1 - y_ratio_X - c_ratio_X) * rng.multivariate_normal([0] * n_features, cov_X, n).T
    return y, c, X.T
