import numpy as np


def simulate_y_c_yhat(ts_ratio_y, cs_ratio_y,
                      ts_ratio_c, cs_ratio_c,
                      ts_ratio_yhat, cs_ratio_yhat,
                      n=100, random_state=None):
    rng = np.random.default_rng(random_state)
    # true signal
    ts = rng.normal(0, 1, n)

    # conf signal
    cs = rng.normal(0, 1, n)

    # c = cs+noise
    c = ts_ratio_c * ts + cs_ratio_c * cs + (1 - ts_ratio_c - cs_ratio_c) * rng.normal(0, 1, n)

    # y = ts+cs+noise
    y = ts_ratio_y * ts + cs_ratio_y * cs + (1 - ts_ratio_y - cs_ratio_y) * rng.normal(0, 1, n)

    # yhat = cs+ts+noise
    yhat = ts_ratio_yhat * ts + cs_ratio_yhat * cs + (1 - ts_ratio_yhat - cs_ratio_yhat) * rng.normal(0, 1, n)

    return y, c, yhat


def _create_covariance_matrix(rho, p):
    a = np.repeat(np.arange(p) + 1, p).reshape(p, p)
    b = np.repeat(np.arange(p) + 1, p).reshape(p, p).T
    return rho ** abs(a - b)


def simulate_y_c_X(ts_ratio_y, cs_ratio_y,
                   ts_ratio_c, cs_ratio_c,
                   ts_ratio_X, cs_ratio_X,
                   n_features, X_corr,
                   dirichlet_sparsity=1,
                   n=100, random_state=None):

    rng = np.random.default_rng(random_state)
    # true signal

    ts = rng.normal(0, 1, n)

    # conf signal
    cs = rng.normal(0, 1, n)

    # c = cs+noise
    c = ts_ratio_c * ts + cs_ratio_c * cs + (1 - ts_ratio_c - cs_ratio_c) * rng.normal(0, 1, n)

    # y = ts+cs+noise
    y = ts_ratio_y * ts + cs_ratio_y * cs + (1 - ts_ratio_y - cs_ratio_y) * rng.normal(0, 1, n)

    cov = _create_covariance_matrix(X_corr, n_features)
    X = ts_ratio_X * ts * \
        (rng.dirichlet([dirichlet_sparsity] * n_features, 1) * (rng.binomial(1, 0.5, n_features)*2-1)).T
    X += cs_ratio_X * cs * \
        (rng.dirichlet([dirichlet_sparsity] * n_features, 1) * (rng.binomial(1, 0.5, n_features)*2-1)).T
    X += (1 - ts_ratio_X - cs_ratio_X) * rng.multivariate_normal([0] * n_features, cov, n).T
    return y, c, X.T
