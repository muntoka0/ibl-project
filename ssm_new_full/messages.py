import numpy as np

from .util import LOG_EPS, logsumexp


def _expand_log_transition(log_transition, T):
    log_transition = np.asarray(log_transition, dtype=float)

    if log_transition.ndim == 2:
        return np.broadcast_to(log_transition[None, :, :], (T - 1,) + log_transition.shape)

    if log_transition.ndim == 3:
        if log_transition.shape[0] == 1:
            return np.broadcast_to(log_transition, (T - 1,) + log_transition.shape[1:])
        if log_transition.shape[0] == T - 1:
            return log_transition

    raise ValueError("log_transition must have shape (K, K), (1, K, K), or (T-1, K, K).")


def forward_backward(log_initial, log_transition, log_likelihoods):
    log_initial = np.asarray(log_initial, dtype=float)
    log_likelihoods = np.asarray(log_likelihoods, dtype=float)

    T, K = log_likelihoods.shape
    log_transition = _expand_log_transition(log_transition, T)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_initial + log_likelihoods[0]

    for t in range(1, T):
        log_alpha[t] = log_likelihoods[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_transition[t - 1],
            axis=0,
        )

    normalizer = float(logsumexp(log_alpha[-1], axis=0))

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_transition[t] + log_likelihoods[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )

    log_expected_states = log_alpha + log_beta - normalizer
    expected_states = np.exp(log_expected_states)
    expected_states /= np.clip(expected_states.sum(axis=1, keepdims=True), LOG_EPS, np.inf)

    expected_joints = np.zeros((max(T - 1, 0), K, K))
    for t in range(T - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_transition[t]
            + log_likelihoods[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - normalizer
        )
        xi = np.exp(log_xi)
        expected_joints[t] = xi / np.clip(xi.sum(), LOG_EPS, np.inf)

    return expected_states, expected_joints, normalizer


def hmm_expected_states(log_pi0, log_Ps, ll):
    """Original-SSM-style alias for forward-backward expectations."""
    return forward_backward(log_pi0, log_Ps, ll)


def hmm_filter(log_pi0, log_Ps, ll):
    """Return filtered state probabilities p(z_t | x_1:t)."""
    log_pi0 = np.asarray(log_pi0, dtype=float)
    ll = np.asarray(ll, dtype=float)
    T, K = ll.shape
    log_Ps = _expand_log_transition(log_Ps, T)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = log_pi0 + ll[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    for t in range(1, T):
        log_alpha[t] = ll[t] + logsumexp(log_alpha[t - 1][:, None] + log_Ps[t - 1], axis=0)
        log_alpha[t] -= logsumexp(log_alpha[t])

    return np.exp(log_alpha)


def hmm_sample(log_pi0, log_Ps, ll, rng=None):
    """Sample a state path from the posterior marginals as a lightweight fallback."""
    rng = np.random.default_rng() if rng is None else rng
    expected_states, _, _ = forward_backward(log_pi0, log_Ps, ll)
    return np.array([
        rng.choice(expected_states.shape[1], p=expected_states[t])
        for t in range(expected_states.shape[0])
    ])


def forward_pass(pi0, Ps, ll, alphas=None):
    """Compatibility helper approximating original ssm.messages.forward_pass."""
    pi0 = np.asarray(pi0, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    ll = np.asarray(ll, dtype=float)
    log_pi0 = np.log(np.clip(pi0, LOG_EPS, 1.0))
    log_Ps = np.log(np.clip(Ps, LOG_EPS, 1.0))
    T, K = ll.shape
    if alphas is None:
        alphas = np.zeros((T, K))
    alphas[0] = log_pi0 + ll[0]
    for t in range(1, T):
        alphas[t] = ll[t] + logsumexp(alphas[t - 1][:, None] + log_Ps, axis=0)
    return alphas


def grad_hmm_normalizer(*args, **kwargs):
    raise NotImplementedError(
        "grad_hmm_normalizer is part of the original autograd/Cython backend and is not implemented in ssm_new_full."
    )


def viterbi(log_initial, log_transition, log_likelihoods):
    log_initial = np.asarray(log_initial, dtype=float)
    log_likelihoods = np.asarray(log_likelihoods, dtype=float)

    T, K = log_likelihoods.shape
    log_transition = _expand_log_transition(log_transition, T)

    delta = np.zeros((T, K))
    backpointers = np.zeros((T, K), dtype=int)

    delta[0] = log_initial + log_likelihoods[0]

    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_transition[t - 1]
        backpointers[t] = np.argmax(scores, axis=0)
        delta[t] = log_likelihoods[t] + np.max(scores, axis=0)

    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))

    for t in range(T - 2, -1, -1):
        states[t] = backpointers[t + 1, states[t + 1]]

    return states
