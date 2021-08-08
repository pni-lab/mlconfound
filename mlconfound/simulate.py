import numpy as np


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

    See Also
    --------
    simulate_y_c_yhat

    References
    ----------
    [1] Jones, M. C. and Pewsey A. (2009). Sinh-arcsinh distributions. Biometrika 96: 761–780
    """
    return np.sinh(delta * np.arcsinh(x) - epsilon)


def simulate_y_c_yhat(cov_y_c,
                      y_ratio_yhat, c_ratio_yhat,
                      n, random_state=None):
    """
    Simulate normally distributed target (y), confounder (c) and predictions (yhat).

    Parameters
    ----------
    cov_y_c: float
        The covariance of y and c.
    y_ratio_yhat: float
        The weight of y in yhat.
    c_ratio_yhat: float
        The weight of c in yhat. Set it to zero for H0.
    n: int
        Number of observations.
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
    [ 0.36959213 -1.16147869  2.34335464] [-0.86093366 -0.04858751  0.80259507] [-0.02156406 -0.43175879  0.61910925]

    """
    rng = np.random.default_rng(random_state)

    y, c = rng.multivariate_normal([0, 0], [[1, cov_y_c], [cov_y_c, 1]], n).T

    yhat = y_ratio_yhat * y + c_ratio_yhat * c + (1 - y_ratio_yhat - c_ratio_yhat) * rng.normal(0, 1, n)

    return y, c, yhat


def _create_covariance_matrix(rho, p):
    a = np.repeat(np.arange(p) + 1, p).reshape(p, p)
    b = np.repeat(np.arange(p) + 1, p).reshape(p, p).T
    return rho ** abs(a - b)


def simulate_y_c_X(cov_y_c,
                   y_ratio_X, c_ratio_X,
                   n_features, X_corr,
                   dirichlet_sparsity,
                   n, random_state=None):
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
