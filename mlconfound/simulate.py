import numpy as np
from warnings import warn


def _scale(x):
    return (x - x.mean()) / x.std()


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


def simulate_y_c_yhat(w_yc,
                      w_yyhat, w_cyhat,
                      n,
                      delta=1,
                      epsilon=0,
                      nonlin_trf_fun=identity,
                      bin_y=False,
                      bin_c=False,
                      bin_yhat=False,
                      scale=True,
                      random_state=None):
    """
    Simulate normally distributed target (y), confounder (c) and predictions (yhat).

    Notes
    -----
    .. math:: y \sim \mathcal{N}(0, 1)

    .. math:: c | y_i \sim \mathcal{N}(w_{y,c} f(y_i), 1)

    .. math:: \hat{y} | y_i, c_i \sim sinh(\delta sinh^{-1}( \mathcal{N}(w_{y,c} f(y_i), 1))-\epsilon)

    Parameters
    ----------
    w_yc: float
        The weight of y in c.
    w_yyhat: float
        The weight of y in yhat.
    w_cyhat: float
        The weight of c in yhat. Set it to zero for H0.
    n: int
        Number of observations.
    delta: float
        The delta param of the sinh_archsin transformation on y's contribution in yhat (only affects yhat).
    epsilon: float
        The epsilon param of the sinh_archsin transformation on y's contribution in yhat (only affects yhat).
    nonlin_trf_fun: callable
        Callable to introduce non-linearity in the conditional distributions. (default: no non-linearity).
    scale: bool
        Scale variables to zero mean and unit variance.
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
    [ 0.30471708 -1.03998411  0.7504512 ] [ 1.32193037 -1.09613765 -0.22579272] [ 1.03955979 -1.35013318  0.31057339]

    """
    rng = np.random.default_rng(random_state)
    y = rng.normal(0, 1, n)

    if bin_y:
        y = (y > 0).astype(int)

    c = np.array([sinh_arcsinh(rng.normal(0, 1, 1), delta=delta, epsilon=epsilon) + w_yc * nonlin_trf_fun(yi) for yi in y]).flatten()
    if scale:
        c = _scale(c)
    if bin_c:
        c = (c > 0).astype(int)

    yhat = np.array([sinh_arcsinh(rng.normal(0, 1, 1), delta=delta, epsilon=epsilon) + w_yyhat * nonlin_trf_fun(yi) + w_cyhat * nonlin_trf_fun(ci)
                                  for yi, ci in zip(y, c)]).flatten()
    if scale:
        yhat = _scale(yhat)
    if bin_yhat:
        yhat = (yhat > 0).astype(int)

    return y, c, yhat
