import numpy as np


def normal_to_heavytailed(x, h=0.2):
    return x*np.exp((h*x**2)/2)


def normal_to_heavytailed_skewed(x, h=0.1, g=0.2):
    return ((np.exp(g*x)-1)/g)*np.exp((h*x**2)/2)


def simulate_y_c_yhat(cov_y_c,
                      y_ratio_yhat, c_ratio_yhat,
                      n, random_state=None):
    rng = np.random.default_rng(random_state)

    y, c = rng.multivariate_normal([0, 0], [[1, cov_y_c], [cov_y_c, 1]], n).T

    yhat = y_ratio_yhat * y + c_ratio_yhat * c  +  (1 - y_ratio_yhat - c_ratio_yhat) * rng.normal(0, 1, n)

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
