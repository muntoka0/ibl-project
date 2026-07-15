import numpy as np
from scipy.special import gammaln

from .util import one_hot
from .util import LOG_EPS, logsumexp


def flatten_to_dim(X, d):
    X = np.asarray(X)
    if X.ndim < d:
        raise ValueError("X must have at least d dimensions.")
    return np.reshape(X[None, ...], (-1,) + X.shape[-d:])


def categorical_logpdf(data, log_probs, mask=None):
    data = np.asarray(data, dtype=int)
    log_probs = np.asarray(log_probs, dtype=float)
    if data.ndim == log_probs.ndim - 1:
        data = data[..., None]
    gathered = np.take_along_axis(log_probs, data, axis=-1).squeeze(axis=-1)
    if mask is not None:
        gathered = np.where(mask.squeeze(axis=-1), gathered, 0.0)
    return gathered


def bernoulli_logpdf(data, logit_p, mask=None):
    data = np.asarray(data)
    logit_p = np.asarray(logit_p)
    log_p1 = -np.logaddexp(0, -logit_p)
    log_p0 = -np.logaddexp(0, logit_p)
    out = data * log_p1 + (1 - data) * log_p0
    if mask is not None:
        out = np.where(mask, out, 0.0)
    return out


def multivariate_normal_logpdf(data, mus, Sigmas=None):
    data = np.asarray(data)
    mus = np.asarray(mus)
    if Sigmas is None:
        return -0.5 * np.sum((data - mus) ** 2, axis=-1)
    Sigmas = np.asarray(Sigmas)
    inv = np.linalg.inv(Sigmas)
    diff = data - mus
    quad = np.einsum("...i,...ij,...j->...", diff, inv, diff)
    logdet = np.linalg.slogdet(Sigmas)[1]
    D = data.shape[-1]
    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def independent_studentst_logpdf(data, mus, sigmas, nus, mask=None):
    data = np.asarray(data)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)
    nus = np.asarray(nus)
    z = (data - mus) / np.clip(sigmas, LOG_EPS, np.inf)
    out = (
        gammaln((nus + 1) / 2)
        - gammaln(nus / 2)
        - 0.5 * np.log(nus * np.pi)
        - np.log(np.clip(sigmas, LOG_EPS, np.inf))
        - ((nus + 1) / 2) * np.log1p((z ** 2) / nus)
    )
    if mask is not None:
        out = np.where(mask, out, 0.0)
    return out


def expected_multinomial_counts(logits):
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
