import numpy as np

from .util import LOG_EPS, normalize


class StationaryTransitions:
    def __init__(self, K, D, M=0, stay_probability=0.95, random_state=0):
        self.K = K
        self.D = D
        self.M = M

        if K == 1:
            transition_matrix = np.ones((1, 1))
        else:
            off_diagonal = (1.0 - stay_probability) / (K - 1)
            transition_matrix = np.full((K, K), off_diagonal)
            np.fill_diagonal(transition_matrix, stay_probability)

            rng = np.random.default_rng(random_state)
            transition_matrix += 0.01 * rng.random((K, K))
            transition_matrix = normalize(transition_matrix, axis=1)

        self.log_Ps = np.log(np.clip(transition_matrix, LOG_EPS, 1.0))

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = np.asarray(value[0], dtype=float)

    @property
    def transition_matrix(self):
        probabilities = np.exp(self.log_Ps)
        return normalize(probabilities, axis=1)

    def permute(self, perm):
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        return None

    def log_transition_matrices(self, data=None, input=None, mask=None, tag=None):
        return np.log(np.clip(self.transition_matrix, LOG_EPS, 1.0))[None, :, :]

    def m_step(self, expectations, datas=None, inputs=None, masks=None, tags=None, **kwargs):
        expected_joints = np.zeros((self.K, self.K))

        for _, expected_joint, _ in expectations:
            if expected_joint.size:
                expected_joints += expected_joint.sum(axis=0)

        expected_joints += LOG_EPS
        transition_matrix = normalize(expected_joints, axis=1)
        self.log_Ps = np.log(np.clip(transition_matrix, LOG_EPS, 1.0))
