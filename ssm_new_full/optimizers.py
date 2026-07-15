import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_weighted_binary_logistic(inputs, choices, sample_weight, current_weights=None):
    inputs = np.asarray(inputs, dtype=float)
    choices = np.asarray(choices, dtype=int).reshape(-1)
    sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    sample_weight = np.clip(sample_weight, 1e-6, np.inf)

    if np.unique(choices).size < 2 or sample_weight.sum() <= 1e-6:
        if current_weights is None:
            return np.zeros(inputs.shape[1])
        return np.asarray(current_weights, dtype=float)

    model = LogisticRegression(
        fit_intercept=False,
        max_iter=1000,
        solver="lbfgs",
        C=1e6,
    )

    try:
        model.fit(inputs, choices, sample_weight=sample_weight)
    except Exception:
        if current_weights is None:
            return np.zeros(inputs.shape[1])
        return np.asarray(current_weights, dtype=float)

    classes = list(model.classes_)
    if classes == [0, 1]:
        return model.coef_[0]

    if 1 in classes:
        return model.coef_[classes.index(1)]

    if current_weights is None:
        return np.zeros(inputs.shape[1])

    return np.asarray(current_weights, dtype=float)
