import numpy as np

from .init_state_distns import InitialStateDistribution
from .messages import forward_backward, viterbi
from .observations import InputDrivenObservations
from .transitions import StationaryTransitions
from .util import ensure_data, ensure_input, ensure_list_data_and_inputs


class HMM:
    """
    Focused GLM-HMM-compatible subset of ssm.HMM.

    Supported path:
        HMM(K, D, M, observations="input_driven_obs", observation_kwargs={"C": 2})

    Implemented methods:
        fit
        expected_states
        most_likely_states

    Implemented public attributes:
        init_state_distn
        transitions.transition_matrix
        observations.weights
    """

    def __init__(
        self,
        K,
        D,
        M=0,
        init_state_distn=None,
        transitions="standard",
        transition_kwargs=None,
        observations="gaussian",
        observation_kwargs=None,
        random_state=0,
        **kwargs,
    ):
        self.K = K
        self.D = D
        self.M = M
        self.random_state = random_state

        if init_state_distn is None:
            init_state_distn = InitialStateDistribution(K, D, M=M)
        self.init_state_distn = init_state_distn

        transition_kwargs = transition_kwargs or {}
        if isinstance(transitions, str):
            if transitions not in ("standard", "stationary"):
                raise NotImplementedError(
                    "ssm_new_full currently supports only standard/stationary transitions."
                )
            self.transitions = StationaryTransitions(
                K,
                D,
                M=M,
                random_state=random_state,
                **transition_kwargs,
            )
        else:
            self.transitions = transitions

        observation_kwargs = observation_kwargs or {}
        if isinstance(observations, str):
            observations = observations.lower()
            if observations != "input_driven_obs":
                raise NotImplementedError(
                    'ssm_new_full currently supports observations="input_driven_obs" only.'
                )
            self.observations = InputDrivenObservations(
                K,
                D,
                M=M,
                random_state=random_state,
                **observation_kwargs,
            )
        else:
            self.observations = observations

        self.log_likelihood_history_ = []

    @property
    def params(self):
        return (
            self.init_state_distn.params,
            self.transitions.params,
            self.observations.params,
        )

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        datas, inputs = ensure_list_data_and_inputs(datas, inputs, self.M)
        self.init_state_distn.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.transitions.initialize(datas, inputs=inputs, masks=masks, tags=tags)
        self.observations.initialize(datas, inputs=inputs, masks=masks, tags=tags, init_method=init_method)

    def permute(self, perm):
        perm = np.asarray(perm, dtype=int)
        if sorted(perm.tolist()) != list(range(self.K)):
            raise ValueError("perm must be a permutation of states 0..K-1.")

        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def _single_expectation(self, data, input):
        data = ensure_data(data)
        input = ensure_input(input, data.shape[0], self.M)

        log_likelihoods = self.observations.log_likelihoods(data, input, mask=None, tag=None)
        log_transition = self.transitions.log_transition_matrices(data, input, mask=None, tag=None)
        log_initial = self.init_state_distn.log_initial_state_distn

        return forward_backward(log_initial, log_transition, log_likelihoods)

    def fit(
        self,
        data,
        inputs=None,
        input=None,
        method="em",
        num_iters=100,
        initialize=True,
        masks=None,
        tags=None,
        **kwargs,
    ):
        if input is not None and inputs is None:
            inputs = input

        if method.lower() != "em":
            raise NotImplementedError('ssm_new_full currently supports method="em" only.')

        datas, input_list = ensure_list_data_and_inputs(data, inputs, self.M)

        if initialize:
            self.initialize(datas, inputs=input_list, masks=masks, tags=tags)

        log_likelihoods = []

        for _ in range(num_iters):
            expectations = [
                self._single_expectation(data_item, input_item)
                for data_item, input_item in zip(datas, input_list)
            ]

            total_log_likelihood = float(sum(expectation[2] for expectation in expectations))
            log_likelihoods.append(total_log_likelihood)

            self.init_state_distn.m_step(expectations, datas, input_list, masks=masks, tags=tags)
            self.transitions.m_step(expectations, datas, input_list, masks=masks, tags=tags)
            self.observations.m_step(expectations, datas, input_list, masks=masks, tags=tags)

        self.log_likelihood_history_ = log_likelihoods
        return log_likelihoods

    def expected_states(self, data, input=None, inputs=None, mask=None, tag=None):
        if input is None:
            input = inputs
        return self._single_expectation(data, input)

    def most_likely_states(self, data, input=None, inputs=None, mask=None, tag=None):
        if input is None:
            input = inputs

        data = ensure_data(data)
        input = ensure_input(input, data.shape[0], self.M)

        log_likelihoods = self.observations.log_likelihoods(data, input, mask=mask, tag=tag)
        log_transition = self.transitions.log_transition_matrices(data, input, mask=mask, tag=tag)
        log_initial = self.init_state_distn.log_initial_state_distn

        return viterbi(log_initial, log_transition, log_likelihoods)

    def log_likelihood(self, data, input=None, inputs=None, mask=None, tag=None):
        if input is None:
            input = inputs
        return self._single_expectation(data, input)[2]
