import numpy as np

from .optimizers import fit_weighted_binary_logistic
from .util import LOG_EPS, sigmoid


class InputDrivenObservations:
    def __init__(self, K, D, M=0, C=2, prior_mean=0, prior_sigma=1000, random_state=0):
        if D != 1:
            raise NotImplementedError("InputDrivenObservations currently supports D=1 only.")
        if C != 2:
            raise NotImplementedError("This pure Python subset currently supports C=2 only.")

        self.K = K
        self.D = D
        self.M = M
        self.C = C
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma

        rng = np.random.default_rng(random_state)
        self.Wk = 0.05 * rng.standard_normal((K, C - 1, M))

    @property
    def params(self):
        return self.Wk

    @params.setter
    def params(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.K, self.C - 1, self.M):
            raise ValueError(f"weights must have shape {(self.K, self.C - 1, self.M)}.")
        self.Wk = value

    @property
    def weights(self):
        return self.Wk

    @weights.setter
    def weights(self, value):
        self.params = value

    def permute(self, perm):
        self.Wk = self.Wk[perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        return None

    def _choice_one_probability(self, input):
        logits = input @ self.Wk[:, 0, :].T
        return sigmoid(logits)

    def calculate_logits(self, input):
        probabilities_one = np.clip(self._choice_one_probability(input), LOG_EPS, 1.0 - LOG_EPS)
        probabilities_zero = 1.0 - probabilities_one

        log_probs = np.zeros((input.shape[0], self.K, self.C))
        log_probs[:, :, 0] = np.log(probabilities_zero)
        log_probs[:, :, 1] = np.log(probabilities_one)
        return log_probs

    def log_likelihoods(self, data, input, mask=None, tag=None):
        choices = np.asarray(data, dtype=int).reshape(-1)

        if np.any((choices < 0) | (choices >= self.C)):
            raise ValueError(f"choices must be integers from 0 to {self.C - 1}.")

        log_probs = self.calculate_logits(input)
        return log_probs[np.arange(len(choices)), :, choices]

    def m_step(self, expectations, datas, inputs, masks=None, tags=None, **kwargs):
        all_inputs = np.vstack(inputs)
        all_choices = np.concatenate([np.asarray(data).reshape(-1).astype(int) for data in datas])
        all_expected_states = np.vstack([expected[0] for expected in expectations])

        for state in range(self.K):
            self.Wk[state, 0, :] = fit_weighted_binary_logistic(
                all_inputs,
                all_choices,
                all_expected_states[:, state],
                current_weights=self.Wk[state, 0, :],
            )

    def sample_x(self, z, xhist=None, input=None, tag=None, with_noise=True):
        if input is None:
            raise ValueError("input is required for InputDrivenObservations.sample_x.")

        input = np.asarray(input, dtype=float)
        if input.ndim == 1:
            input = input[None, :]

        z = np.asarray(z, dtype=int)
        if z.ndim == 0:
            z = np.repeat(z, input.shape[0])

        probabilities_one = self._choice_one_probability(input)
        rng = np.random.default_rng()
        samples = np.array([
            rng.binomial(1, probabilities_one[t, z[t]])
            for t in range(input.shape[0])
        ])
        return samples[:, None]
