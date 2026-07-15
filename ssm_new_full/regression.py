import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def fit_linear_regression(Xs, ys, weights=None, fit_intercept=True, **kwargs):
    Xs = Xs if isinstance(Xs, (list, tuple)) else [Xs]
    ys = ys if isinstance(ys, (list, tuple)) else [ys]
    X = np.vstack(Xs)
    y = np.vstack([np.asarray(item).reshape(len(item), -1) for item in ys])
    sample_weight = None
    if weights is not None:
        weights = weights if isinstance(weights, (list, tuple)) else [weights]
        sample_weight = np.concatenate([np.asarray(w).reshape(-1) for w in weights])
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y, sample_weight=sample_weight)
    return model.coef_, model.intercept_


def fit_multiclass_logistic_regression(X, y, weights=None, **kwargs):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).reshape(-1)
    model = LogisticRegression(max_iter=1000, multi_class="auto")
    model.fit(X, y, sample_weight=weights)
    return model


def fit_negative_binomial_integer_r(*args, **kwargs):
    raise NotImplementedError(
        "Negative-binomial regression is outside the ssm_new_full GLM-HMM subset."
    )
