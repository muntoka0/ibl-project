"""Ashwood-style mouse-level GLM-HMM helper functions.

These helpers are used by IBL_behavior_dataQ3_Ashwood-style90.ipynb.
They keep the notebook focused on the analysis flow while preserving the
same Ashwood-style model fitting, decoding, and summary logic.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

import ssm_new_full as ssm


@dataclass
class AshwoodFitConfig:
    """Notebook settings needed by the mouse-level helper functions."""

    input_columns: Sequence[str]
    time_sort_column: str
    state_weight_labels: Dict[str, str]
    stim_weight_col: str
    bias_weight_col: str
    num_folds: int
    model_k_list: Sequence[int]
    hmm_max_iter: int
    hmm_n_restarts: int
    stay_probability: float
    prior_sigma: float
    engaged_contrast_min: float
    bias_sign_min: float


def state_parameter_table_from_ssm(model, input_columns):
    """Convert fitted observation weights into a readable per-state table."""
    weights = np.asarray(model.observations.weights)[:, 0, :]
    rows = []
    for state_index in range(model.K):
        row = {'state': state_index}
        for column_name, weight in zip(input_columns, weights[state_index]):
            row[f'{column_name}_weight'] = weight
        rows.append(row)
    return pd.DataFrame(rows)


def session_arrays(mouse_df, sessions, config):
    """Build per-session data arrays so the HMM sees ordered trial sequences."""
    datas, inputs, frames = [], [], []
    session_set = set(sessions)
    for session in pd.unique(mouse_df['session']):
        if session not in session_set:
            continue
        session_df = (
            mouse_df[mouse_df['session'] == session]
            .sort_values(config.time_sort_column)
            .reset_index(drop=True)
            .copy(deep=True)
        )
        if session_df.empty:
            continue
        datas.append(session_df['choice_binary'].to_numpy(dtype=int).reshape(-1, 1))
        inputs.append(session_df[list(config.input_columns)].to_numpy(dtype=float))
        frames.append(session_df)
    return datas, inputs, frames


def make_session_folds(sessions, n_folds=5, seed=65):
    """Assign whole sessions to cross-validation folds."""
    sessions = np.array(list(sessions), dtype=object)
    n_folds = min(n_folds, len(sessions))
    if n_folds < 2:
        return pd.DataFrame({'session': sessions, 'fold': np.zeros(len(sessions), dtype=int)})
    rng = np.random.default_rng(seed)
    folds = np.resize(np.arange(n_folds), len(sessions))
    folds = rng.permutation(folds)
    return pd.DataFrame({'session': sessions, 'fold': folds})


def fit_hmm_model(datas, inputs, n_states, config, random_state=0):
    """Fit a GLM-HMM and keep the best restart by final log likelihood."""
    best_model = None
    best_history = None
    best_ll = -np.inf
    for restart in range(config.hmm_n_restarts):
        model = ssm.HMM(
            K=n_states,
            D=1,
            M=len(config.input_columns),
            observations='input_driven_obs',
            observation_kwargs={'C': 2, 'prior_sigma': config.prior_sigma},
            transitions='standard',
            transition_kwargs={'stay_probability': config.stay_probability},
            random_state=random_state + restart,
        )
        history = model.fit(datas, inputs=inputs, method='em', num_iters=config.hmm_max_iter)
        if history[-1] > best_ll:
            best_model = model
            best_history = history
            best_ll = history[-1]
    return best_model, best_history


def choice_prob_by_state(model, X):
    """Calculate P(choice == 1) for each hidden state."""
    weights = model.observations.weights[:, 0, :]
    logits = np.asarray(X, dtype=float) @ weights.T
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def hmm_loglik_and_accuracy(model, datas, inputs):
    """Score a fitted GLM-HMM on held-out sessions."""
    total_ll = 0.0
    total_n = 0
    correct = []
    for y, X in zip(datas, inputs):
        y_vec = y.reshape(-1).astype(int)
        total_ll += model.log_likelihood(y, input=X)
        total_n += len(y_vec)
        states = model.most_likely_states(y, input=X)
        probs = choice_prob_by_state(model, X)[np.arange(len(y_vec)), states]
        pred = (probs >= 0.5).astype(int)
        correct.extend((pred == y_vec).tolist())
    return total_ll / total_n, float(np.mean(correct)) if correct else np.nan


def glm_loglik_and_accuracy(train_datas, train_inputs, test_datas, test_inputs):
    """Fit and score the 1-state logistic regression baseline."""
    X_train = np.vstack(train_inputs)
    y_train = np.concatenate([y.reshape(-1).astype(int) for y in train_datas])
    X_test = np.vstack(test_inputs)
    y_test = np.concatenate([y.reshape(-1).astype(int) for y in test_datas])
    if len(np.unique(y_train)) < 2:
        raise ValueError('Need both choice classes 0 and 1 in GLM training data.')
    model = LogisticRegression(max_iter=1000, fit_intercept=False)
    model.fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return -log_loss(y_test, p_test, labels=[0, 1]), accuracy_score(y_test, p_test >= 0.5), model


def stimulus_accuracy(df):
    """Measure whether choices matched stimulus side on nonzero-contrast trials."""
    if df.empty:
        return np.nan
    mask = df['signed_contrast'] != 0
    if mask.sum() == 0:
        return np.nan
    correct = (
        ((df.loc[mask, 'signed_contrast'] > 0) & (df.loc[mask, 'choice_binary'] == 1))
        | ((df.loc[mask, 'signed_contrast'] < 0) & (df.loc[mask, 'choice_binary'] == 0))
    )
    return correct.mean()


def ashwood_state_order(params_df, config):
    """Assign engaged, left-biased, and right-biased states by Ashwood-style weights."""
    stim_col = config.state_weight_labels['stimulus']
    bias_col = config.state_weight_labels['bias']
    engaged_state = int(params_df.loc[params_df[stim_col].idxmax(), 'state'])

    reduced = params_df.copy(deep=True)
    max_bias = reduced[bias_col].max()
    reduced.loc[reduced['state'] == engaged_state, bias_col] = max_bias - 0.001
    left_state = int(reduced.loc[reduced[bias_col].idxmin(), 'state'])

    remaining = [int(state) for state in params_df['state'] if int(state) not in {engaged_state, left_state}]
    right_state = remaining[0] if remaining else int(params_df.loc[params_df[bias_col].idxmax(), 'state'])
    return engaged_state, left_state, right_state


def make_state_weight_table(params_df, subject, engaged_state, left_state, right_state, config):
    """Build the readable state-weight table used for Ashwood interpretation."""
    rename_map = {
        config.state_weight_labels['stimulus']: 'signed_contrast_weight',
        config.state_weight_labels['previous_choice']: 'prev_choice_centered_weight',
        config.state_weight_labels['wsls']: 'prev_choice_x_reward_weight',
        config.state_weight_labels['bias']: 'bias_weight',
    }
    readable = params_df.rename(columns=rename_map).copy(deep=True)

    label_by_state = {
        engaged_state: 'stimulus-sensitive / engaged',
        left_state: 'left-biased / disengaged',
        right_state: 'right-biased / disengaged',
    }
    weight_cols = [
        'signed_contrast_weight',
        'bias_weight',
        'prev_choice_centered_weight',
        'prev_choice_x_reward_weight',
    ]

    rows = []
    for _, row in readable.iterrows():
        signed_values = row[weight_cols].astype(float)
        strongest_weight = signed_values.abs().idxmax().replace('_weight', '')
        rows.append({
            'subject': subject,
            'state': int(row['state']),
            'suggested_label': label_by_state.get(int(row['state']), 'other'),
            'strongest_weight': strongest_weight,
            'strongest_weight_value': signed_values.loc[strongest_weight + '_weight'],
            'signed_contrast_weight': row['signed_contrast_weight'],
            'bias_weight': row['bias_weight'],
            'prev_choice_centered_weight': row['prev_choice_centered_weight'],
            'prev_choice_x_reward_weight': row['prev_choice_x_reward_weight'],
        })
    return pd.DataFrame(rows)


def safe_dwell(p_self):
    """Convert self-transition probability into expected dwell time."""
    if pd.isna(p_self):
        return np.nan
    if p_self >= 1:
        return np.inf
    return 1 / (1 - p_self)


def prepare_mouse_for_fit(subject, mouse_idx, baseline_eligible_df, config):
    """Select one mouse's ordered baseline trials and create session folds."""
    mouse_df = (
        baseline_eligible_df[baseline_eligible_df['subject'] == subject]
        .sort_values(['session', config.time_sort_column])
        .copy(deep=True)
    )
    sessions = pd.unique(mouse_df['session']).tolist()
    n_sessions = len(sessions)
    n_trials = len(mouse_df)
    fold_lookup = make_session_folds(sessions, config.num_folds, seed=65 + mouse_idx)
    return mouse_df, sessions, n_sessions, n_trials, fold_lookup


def compare_models_for_mouse(subject, mouse_idx, mouse_df, fold_lookup, config):
    """Compare the 1-state GLM against requested GLM-HMM models."""
    comparison_rows = []

    for fold in sorted(fold_lookup['fold'].unique()):
        train_sessions = fold_lookup.loc[fold_lookup['fold'] != fold, 'session'].tolist()
        test_sessions = fold_lookup.loc[fold_lookup['fold'] == fold, 'session'].tolist()
        train_datas, train_inputs, _ = session_arrays(mouse_df, train_sessions, config)
        test_datas, test_inputs, _ = session_arrays(mouse_df, test_sessions, config)

        if len(train_datas) == 0 or len(test_datas) == 0:
            continue

        glm_test_ll, glm_test_acc, _ = glm_loglik_and_accuracy(
            train_datas, train_inputs, test_datas, test_inputs
        )
        comparison_rows.append({
            'subject': subject,
            'fold': int(fold),
            'model': '1-state GLM',
            'n_states': 1,
            'test_log_likelihood_per_trial': glm_test_ll,
            'test_accuracy': glm_test_acc,
        })

        for k in [state for state in config.model_k_list if state > 1]:
            model, history = fit_hmm_model(
                train_datas,
                train_inputs,
                n_states=k,
                config=config,
                random_state=1000 * mouse_idx + 10 * int(fold),
            )
            test_ll, test_acc = hmm_loglik_and_accuracy(model, test_datas, test_inputs)
            comparison_rows.append({
                'subject': subject,
                'fold': int(fold),
                'model': f'{k}-state GLM-HMM',
                'n_states': k,
                'test_log_likelihood_per_trial': test_ll,
                'test_accuracy': test_acc,
            })

    return comparison_rows


def fit_final_three_state_model(mouse_idx, mouse_df, sessions, config):
    """Fit the final Ashwood-style 3-state GLM-HMM for one mouse."""
    all_datas, all_inputs, all_frames = session_arrays(mouse_df, sessions, config)
    if len(all_datas) == 0:
        raise ValueError('Need at least one session for mouse-level fit.')

    all_y = np.concatenate([y.reshape(-1).astype(int) for y in all_datas])
    if len(np.unique(all_y)) < 2:
        raise ValueError('Need both choice classes 0 and 1 for mouse-level fit.')

    model3, history3 = fit_hmm_model(
        all_datas,
        all_inputs,
        n_states=3,
        config=config,
        random_state=5000 + mouse_idx,
    )
    params3 = state_parameter_table_from_ssm(model3, config.input_columns)
    transition = model3.transitions.transition_matrix
    engaged_state, left_state, right_state = ashwood_state_order(params3, config)

    return {
        'model3': model3,
        'history3': history3,
        'params3': params3,
        'transition': transition,
        'engaged_state': engaged_state,
        'left_state': left_state,
        'right_state': right_state,
        'all_datas': all_datas,
        'all_inputs': all_inputs,
        'all_frames': all_frames,
    }


def decode_mouse_trials(final_fit):
    """Decode the most likely state and posterior state probabilities for every trial."""
    decoded_mouse_parts = []
    model3 = final_fit['model3']
    engaged_state = final_fit['engaged_state']
    left_state = final_fit['left_state']
    right_state = final_fit['right_state']

    for session_df, y, X in zip(final_fit['all_frames'], final_fit['all_datas'], final_fit['all_inputs']):
        states = model3.most_likely_states(y, input=X)
        posterior, _, _ = model3.expected_states(y, input=X)

        decoded = session_df.copy(deep=True)
        decoded['most_likely_state'] = states
        decoded['engaged_state'] = engaged_state
        decoded['left_biased_state'] = left_state
        decoded['right_biased_state'] = right_state

        for state_idx in range(model3.K):
            decoded[f'state_{state_idx}_probability'] = posterior[:, state_idx]

        decoded_mouse_parts.append(decoded)

    return pd.concat(decoded_mouse_parts, ignore_index=True)


def summarize_mouse_fit(subject, final_fit, decoded_mouse_df, config):
    """Summarize state interpretation, accuracy, occupancy, switching, and dwell time."""
    params3 = final_fit['params3']
    transition = final_fit['transition']
    engaged_state = final_fit['engaged_state']
    left_state = final_fit['left_state']
    right_state = final_fit['right_state']

    max_stimulus_weight = params3[config.stim_weight_col].max()
    min_bias_weight = params3[config.bias_weight_col].min()
    max_bias_weight = params3[config.bias_weight_col].max()
    engaged_found = max_stimulus_weight > config.engaged_contrast_min
    left_found = params3.loc[params3['state'] == left_state, config.bias_weight_col].iloc[0] < -config.bias_sign_min
    right_found = params3.loc[params3['state'] == right_state, config.bias_weight_col].iloc[0] > config.bias_sign_min
    ashwood_state_order_complete = len({engaged_state, left_state, right_state}) == 3

    state_row = {
        'subject': subject,
        'engaged_state': engaged_state,
        'left_biased_state': left_state,
        'right_biased_state': right_state,
        'engaged_state_found': engaged_found,
        'left_biased_state_found': left_found,
        'right_biased_state_found': right_found,
        'ashwood_state_order_complete': ashwood_state_order_complete,
        'all_3_state_types_found': ashwood_state_order_complete and engaged_found and left_found and right_found,
        'max_stimulus_weight': max_stimulus_weight,
        'min_bias_weight': min_bias_weight,
        'max_bias_weight': max_bias_weight,
        'engaged_bias_weight': params3.loc[params3['state'] == engaged_state, config.bias_weight_col].iloc[0],
        'left_biased_bias_weight': params3.loc[params3['state'] == left_state, config.bias_weight_col].iloc[0],
        'right_biased_bias_weight': params3.loc[params3['state'] == right_state, config.bias_weight_col].iloc[0],
    }

    state_weight_df = make_state_weight_table(params3, subject, engaged_state, left_state, right_state, config)

    accuracy_row = {
        'subject': subject,
        'accuracy_engaged': stimulus_accuracy(decoded_mouse_df[decoded_mouse_df['most_likely_state'] == engaged_state]),
        'accuracy_left_biased': stimulus_accuracy(decoded_mouse_df[decoded_mouse_df['most_likely_state'] == left_state]),
        'accuracy_right_biased': stimulus_accuracy(decoded_mouse_df[decoded_mouse_df['most_likely_state'] == right_state]),
        'overall_accuracy': stimulus_accuracy(decoded_mouse_df),
    }

    occupancy_row = {
        'subject': subject,
        'engaged_fraction': (decoded_mouse_df['most_likely_state'] == engaged_state).mean(),
        'left_biased_fraction': (decoded_mouse_df['most_likely_state'] == left_state).mean(),
        'right_biased_fraction': (decoded_mouse_df['most_likely_state'] == right_state).mean(),
    }

    session_switches = []
    for session, session_decoded in decoded_mouse_df.groupby('session'):
        n_switches = int(
            session_decoded['most_likely_state']
            .ne(session_decoded['most_likely_state'].shift(1))
            .sum()
            - 1
        )
        session_switches.append(max(n_switches, 0))

    switch_row = {
        'subject': subject,
        'fraction_sessions_with_any_switch': np.mean(np.array(session_switches) > 0),
        'fraction_sessions_with_multiple_switches': np.mean(np.array(session_switches) > 1),
        'median_switches_per_session': np.median(session_switches),
    }

    p_engaged = transition[engaged_state, engaged_state]
    p_left = transition[left_state, left_state]
    p_right = transition[right_state, right_state]
    dwell_row = {
        'subject': subject,
        'mean_self_transition_probability': np.mean(np.diag(transition)),
        'expected_dwell_time_engaged': safe_dwell(p_engaged),
        'expected_dwell_time_left_biased': safe_dwell(p_left),
        'expected_dwell_time_right_biased': safe_dwell(p_right),
    }

    return state_row, state_weight_df, accuracy_row, occupancy_row, switch_row, dwell_row


__all__ = [
    'AshwoodFitConfig',
    'state_parameter_table_from_ssm',
    'session_arrays',
    'make_session_folds',
    'fit_hmm_model',
    'choice_prob_by_state',
    'hmm_loglik_and_accuracy',
    'glm_loglik_and_accuracy',
    'stimulus_accuracy',
    'ashwood_state_order',
    'make_state_weight_table',
    'safe_dwell',
    'prepare_mouse_for_fit',
    'compare_models_for_mouse',
    'fit_final_three_state_model',
    'decode_mouse_trials',
    'summarize_mouse_fit',
]
