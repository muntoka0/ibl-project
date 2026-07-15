import numpy as np

from .util import logsumexp, normalize


class InitialStateDistribution:
    def __init__(self, K, D, M=0):
        self.K = K
        self.D = D
        self.M = M
        self.log_pi0 = -np.log(K) * np.ones(K)

    @property
    def params(self):
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        self.log_pi0 = np.asarray(value[0], dtype=float)

    @property
    def initial_state_distn(self):
        return np.exp(self.log_initial_state_distn)

    @property
    def log_initial_state_distn(self):
        return self.log_pi0 - logsumexp(self.log_pi0)

    def permute(self, perm):
        self.log_pi0 = self.log_pi0[perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        return None

    def m_step(self, expectations, datas=None, inputs=None, masks=None, tags=None, **kwargs):
        pi0 = np.zeros(self.K)

        for expected_states, _, _ in expectations:
            pi0 += expected_states[0]

        pi0 = normalize(pi0 + 1e-8)
        self.log_pi0 = np.log(pi0)
