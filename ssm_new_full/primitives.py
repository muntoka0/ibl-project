import numpy as np

from .messages import forward_backward
from .util import LOG_EPS


def hmm_normalizer(pi0, Ps, ll):
    pi0 = np.asarray(pi0, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    ll = np.asarray(ll, dtype=float)
    log_pi0 = np.log(np.clip(pi0, LOG_EPS, 1.0))
    log_Ps = np.log(np.clip(Ps, LOG_EPS, 1.0))
    return forward_backward(log_pi0, log_Ps, ll)[2]


def lds_log_probability(*args, **kwargs):
    raise NotImplementedError("LDS primitives are outside the ssm_new_full GLM-HMM subset.")


def lds_sample(*args, **kwargs):
    raise NotImplementedError("LDS primitives are outside the ssm_new_full GLM-HMM subset.")


def lds_mean(*args, **kwargs):
    raise NotImplementedError("LDS primitives are outside the ssm_new_full GLM-HMM subset.")
