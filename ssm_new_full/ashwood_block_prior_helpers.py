"""Helper functions for Ashwood block-prior vs disengaged-bias checks."""

import numpy as np


def prepare_trial_order(all_trials):
    """Sort trials and add trial number within each session."""
    if 'session' not in all_trials.columns and 'eid' in all_trials.columns:
        all_trials['session'] = all_trials['eid']

    all_trials = all_trials.sort_values(
        ['subject', 'session', 'intervals_0']
    ).copy()

    all_trials['ashwood_trial_number'] = (
        all_trials.groupby(['subject', 'session']).cumcount() + 1
    )

    return all_trials


def add_choice_and_stimulus_columns(all_trials):
    """Add binary choice, signed contrast, and GLM bias column."""
    all_trials['bias'] = 1.0

    if 'choice_binary' not in all_trials.columns:
        all_trials['choice_binary'] = np.select(
            [all_trials['choice'].eq(-1), all_trials['choice'].eq(1)],
            [0, 1],
            default=np.nan,
        )

    if 'signed_contrast' not in all_trials.columns:
        all_trials['signed_contrast'] = (
            all_trials['contrastRight'].fillna(0)
            - all_trials['contrastLeft'].fillna(0)
        )

    return all_trials


def add_history_regressors(all_trials):
    """Add Ashwood history regressors: previous choice and win-stay / lose-switch."""
    if 'prev_choice_binary' not in all_trials.columns:
        all_trials['prev_choice_binary'] = (
            all_trials.groupby(['subject', 'session'])['choice_binary'].shift(1)
        )

    if 'prev_rewarded' not in all_trials.columns:
        rewarded = all_trials['feedbackType'].eq(1).astype(float)
        all_trials['prev_rewarded'] = (
            rewarded.groupby([all_trials['subject'], all_trials['session']]).shift(1)
        )

    all_trials['prev_choice_centered'] = (
        all_trials['prev_choice_binary'].astype(float) * 2 - 1
    )

    all_trials['prev_reward_centered'] = (
        all_trials['prev_rewarded'].astype(float) * 2 - 1
    )

    all_trials['prev_choice_x_reward'] = (
        all_trials['prev_choice_centered']
        * all_trials['prev_reward_centered']
    )

    return all_trials


def add_state_labels(trial_df):
    """Label each decoded trial as engaged, left-biased, right-biased, or other."""
    trial_df['state_label'] = np.select(
        [
            trial_df['most_likely_state'].eq(trial_df['engaged_state']),
            trial_df['most_likely_state'].eq(trial_df['left_biased_state']),
            trial_df['most_likely_state'].eq(trial_df['right_biased_state']),
        ],
        ['engaged', 'left_biased', 'right_biased'],
        default='other',
    )

    return trial_df


def add_accuracy_columns(trial_df):
    """Check whether the mouse chose the stimulus side on non-zero contrast trials."""
    trial_df['abs_contrast'] = trial_df['signed_contrast'].abs()

    trial_df['stimulus_correct'] = np.where(
        trial_df['signed_contrast'].ne(0),
        (
            (trial_df['signed_contrast'].gt(0) & trial_df['choice_binary'].eq(1))
            | (trial_df['signed_contrast'].lt(0) & trial_df['choice_binary'].eq(0))
        ),
        np.nan,
    )

    trial_df['stimulus_error'] = np.where(
        trial_df['signed_contrast'].ne(0),
        ~trial_df['stimulus_correct'].astype(bool),
        np.nan,
    )

    return trial_df


def add_high_probability_side_choice(trial_df):
    """Check whether each choice goes toward the high-probability side of that block."""
    trial_df = trial_df.drop(
        columns=[
            'fraction_positive_signed_contrast',
            'high_probability_choice_binary',
            'fraction_positive_signed_contrast_x',
            'fraction_positive_signed_contrast_y',
            'high_probability_choice_binary_x',
            'high_probability_choice_binary_y',
            'choice_to_high_probability_side',
        ],
        errors='ignore',
    ).copy()

    side_df = (
        trial_df[trial_df['signed_contrast'].ne(0)]
        .groupby(['analysis_label', 'block_type'], as_index=False)
        .agg(
            fraction_positive_signed_contrast=(
                'signed_contrast',
                lambda x: x.gt(0).mean(),
            )
        )
    )

    side_df['high_probability_choice_binary'] = np.where(
        side_df['fraction_positive_signed_contrast'].ge(0.5),
        1,
        0,
    )

    trial_df = trial_df.merge(
        side_df,
        on=['analysis_label', 'block_type'],
        how='left',
    )

    trial_df['choice_to_high_probability_side'] = (
        trial_df['choice_binary'].eq(trial_df['high_probability_choice_binary'])
    )

    return trial_df


def add_contrast_bins(trial_df):
    """Mark low-contrast and high-contrast trials within each block analysis."""
    trial_df = trial_df.drop(
        columns=[
            'low_threshold',
            'high_threshold',
            'low_threshold_x',
            'low_threshold_y',
            'high_threshold_x',
            'high_threshold_y',
            'is_low_contrast',
            'is_high_contrast',
            'low_contrast_correct',
            'high_contrast_error',
        ],
        errors='ignore',
    ).copy()

    thresholds = (
        trial_df[trial_df['abs_contrast'].gt(0)]
        .groupby(['analysis_label', 'block_type'])['abs_contrast']
        .quantile([0.50, 0.75])
        .unstack()
        .rename(columns={0.50: 'low_threshold', 0.75: 'high_threshold'})
        .reset_index()
    )

    trial_df = trial_df.merge(
        thresholds,
        on=['analysis_label', 'block_type'],
        how='left',
    )

    trial_df['is_low_contrast'] = (
        trial_df['abs_contrast'].gt(0)
        & trial_df['abs_contrast'].le(trial_df['low_threshold'])
    )

    trial_df['is_high_contrast'] = (
        trial_df['abs_contrast'].ge(trial_df['high_threshold'])
    )

    trial_df['low_contrast_correct'] = (
        trial_df['stimulus_correct'].where(trial_df['is_low_contrast'])
    )

    trial_df['high_contrast_error'] = (
        trial_df['stimulus_error'].where(trial_df['is_high_contrast'])
    )

    return trial_df


def add_choice_switch_columns(trial_df):
    """Check whether the mouse switched choice from the previous trial."""
    trial_df = trial_df.sort_values(
        ['analysis_label', 'subject', 'session', 'intervals_0']
    ).copy()

    trial_df['previous_choice'] = (
        trial_df.groupby(['analysis_label', 'subject', 'session'])['choice_binary']
        .shift(1)
    )

    trial_df['choice_switch'] = np.where(
        trial_df['previous_choice'].notna(),
        trial_df['choice_binary'].ne(trial_df['previous_choice']),
        np.nan,
    )

    return trial_df


def add_switch_epoch_labels(trial_df):
    """Label trials as early after block switch, late block, or other."""
    trial_df['switch_epoch'] = np.select(
        [
            (
                trial_df['segment_starts_after_switch']
                & trial_df['trials_after_block_switch'].between(1, 10)
            ),
            (
                trial_df['segment_starts_after_switch']
                & trial_df['trials_after_block_switch'].ge(31)
            ),
        ],
        ['early_after_switch', 'late_block'],
        default='other',
    )

    return trial_df

