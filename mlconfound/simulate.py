import numpy as np


def simulate(ts_ratio_y, cs_ratio_y,
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

    return y, yhat, c
