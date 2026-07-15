# IBL behavior latent-strategy project

This project analyzes mouse decision behavior from the International Brain Laboratory behavioral dataset. The notebooks build up from simple psychometric logistic regression to lapse models, history effects, reward/time state variables, GLM-HMM hidden states, Roy/PsyTrack continuous weight dynamics, and a final sex-biased latent-strategy project.

## Main question

Do mice use one stable decision strategy, or do their behavioral strategy and internal state change across trials, sessions, rewards, and task switches?

The project treats behavior as a sequence of increasingly rich models:

1. Fixed psychometric logistic regression.
2. Logistic regression with lapse probability.
3. Logistic regression with previous choice and previous reward history.
4. Sliding-window logistic regression for within-session nonstationarity.
5. Reward/time predictors for satiety or session-progress effects.
6. Hidden-state GLM-HMM models for discrete strategy states.
7. Roy/PsyTrack dynamic GLM for continuous latent changes.
8. Sex-biased latent strategy analysis for set-shift behavior.

## Recommended notebooks

Use these as the current cleaned-up versions:

| File | Purpose | Status |
|---|---|---|
| `IBL_behavior_dataQ1.ipynb` | Baseline psychometric logistic regression across mice | Base analysis |
| `IBL_behavior_dataQ2.ipynb` | Logistic regression plus lapse probability | Base analysis |
| `IBL_behavior_dataQ3.ipynb` | Previous choice/reward history model | Base analysis |
| `IBL_behavior_dataQ4.ipynb` | Sliding-window stationarity analysis | Base analysis |
| `IBL_behavior_dataQ5.ipynb` | Reward fraction and trial fraction internal-state tests | Base analysis |
| `IBL_behavior_dataQ6_ssm-new.ipynb` | Early Python 3.11-compatible hidden-state test | Intermediate |
| `IBL_behavior_dataQ8_2.ipynb` | Current GLM-HMM solution using `ssm_new_full` | Recommended for Q8 |
| `IBL_behavior_dataQ9_3.ipynb` | Current Roy/PsyTrack continuous-weight GLM | Recommended for Q9 |
| `IBL_behavior_dataQ10.ipynb` | Sex-biased latent strategy dynamics project | Current project version |
| `IBL_behavior_sex_latent_strategy_project.ipynb` | Standalone final project notebook | Final project file |

Older versions such as `IBL_behavior_dataQ8.ipynb`, `IBL_behavior_dataQ8_1.ipynb`, `IBL_behavior_dataQ9.ipynb`, `IBL_behavior_dataQ9_1.ipynb`, and `IBL_behavior_dataQ9_2.ipynb` are kept for development history. For the cleanest current analysis, use `Q8_2` and `Q9_3`.

## Model progression

### Q1: Logistic psychometric model

Fits one logistic regression per mouse using signed contrast to predict binary choice. This gives the baseline psychometric curve and baseline log likelihood.

### Q2: Logistic plus lapse

Adds a lapse probability so the model can account for high-contrast mistakes. The model tests whether mice sometimes make stimulus-independent choices even when evidence is easy.

### Q3: Choice autocorrelation

Adds previous choice and previous reward as predictors. This tests whether current choice depends on trial history, not only current sensory contrast.

### Q4: Stationarity

Fits models in sliding windows within a session. This tests whether model parameters are stable or change inside a single session.

### Q5: Reward/time internal state

Adds trial fraction, reward fraction, and interaction terms. This tests whether accumulated reward or session progress changes behavior, possibly reflecting satiety, fatigue, or boredom.

### Q8: GLM-HMM discrete internal states

Uses a GLM-HMM to model discrete hidden states. Each state has its own logistic regression parameters. This follows the Ashwood-style idea that animals may switch among discrete strategies such as engaged, biased, or lapse-like states.

Current recommended file:

```text
IBL_behavior_dataQ8_2.ipynb
```

Current import style:

```python
import ssm_new_full as ssm

model = ssm.HMM(
    K=3,
    D=1,
    M=3,
    observations="input_driven_obs",
    observation_kwargs={"C": 2}
)
```

### Q9: Roy/PsyTrack continuous latent dynamics

Uses the real `psytrack` package to fit a dynamic GLM with smoothly changing weights. This follows the Roy-style idea that internal state may vary continuously rather than jumping between discrete HMM states.

Current recommended file:

```text
IBL_behavior_dataQ9_3.ipynb
```

Key model idea:

```text
P(choice_t = 1) = sigmoid(x_t dot w_t)
```

where `w_t` changes smoothly across trials.

### Q10 / final project: Sex-biased latent strategy dynamics

Tests the hypothesis that sex differences in set-shift performance are driven by latent strategy dynamics rather than only sensory ability or attention. The analysis compares rule-update speed, old-rule perseveration, high-contrast error types, sex interactions in logistic regression, and GLM-HMM state occupancy.

Recommended final files:

```text
IBL_behavior_dataQ10.ipynb
IBL_behavior_sex_latent_strategy_project.ipynb
```

## Local Python modules

### `ssm_new_full/`

`ssm_new_full` is a Python 3.11-compatible GLM-HMM subset inspired by the original `ssm` API. It is not the full original `ssm` package.

Fully working for this project:

```text
__init__.py
hmm.py
observations.py
transitions.py
init_state_distns.py
messages.py
optimizers.py
util.py
```

Supported use case:

```text
binary choices
choices shape = (T, 1)
choices coded as 0/1
inputs shape = (T, M)
observations = "input_driven_obs"
method = "em"
D = 1
C = 2
```

Partially ported compatibility helpers:

```text
stats.py
regression.py
model_selection.py
plots.py
preprocessing.py
primitives.py
hierarchical.py
```

Placeholder or limited files:

```text
lds.py
emissions.py
variational.py
extensions/
```

These are present mostly so imports do not fail. They do not implement the full LDS/SLDS/variational functionality from original `ssm`.

### `ssm_new.py`

A simpler educational GLM-HMM helper used in earlier notebooks. Prefer `ssm_new_full` for current Q8/Q9-related GLM-HMM work.

### `ssm_source/`

Original/reference `ssm` source. This is useful for reading the original API, but it may fail in Python 3.11 because the original package depends on older compiled/Cython backend pieces.

## Environment setup

The notebooks use IBL data tools plus scientific Python packages.

Common dependencies:

```text
ONE-api
ibllib
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
statsmodels
psytrack
jupyter
```

If you are using Anaconda, install into the active environment with:

```bash
python -m pip install ONE-api ibllib numpy pandas matplotlib seaborn scipy scikit-learn statsmodels psytrack jupyter
```

If Jupyter says `ModuleNotFoundError` even after installing a package, install into the notebook kernel itself:

```python
import sys
!{sys.executable} -m pip install psytrack
```

For Q9, `IBL_behavior_dataQ9_3.ipynb` already includes a kernel-aware PsyTrack import/install cell.

## Running order

The notebooks are cumulative. A safe order is:

```text
Q1 -> Q2 -> Q3 -> Q4 -> Q5 -> Q8_2 -> Q9_3 -> Q10/final project
```

Several later notebooks contain copied earlier cells, so they can often run as standalone notebooks from top to bottom. If a later notebook has missing variables, run the earlier preprocessing/model cells in the same notebook first.

## Data notes

The analysis loads IBL behavioral data through ONE/IBL tools. Running the notebooks may require network access on the first run and may cache data locally afterward.

Core columns created across notebooks include:

```text
subject
session
signed_contrast
choice_binary
prev_choice_binary
prev_rewarded
trial_fraction
reward_fraction
```

Later notebooks make deep copies such as `df_q5`, `df_q8_2`, and `df_q9_3` to avoid changing previous results.

## Interpretation map

| Model | Internal-state assumption | Best file |
|---|---|---|
| Q1 logistic regression | One fixed psychometric strategy | `IBL_behavior_dataQ1.ipynb` |
| Q2 lapse model | Random lapses on top of fixed strategy | `IBL_behavior_dataQ2.ipynb` |
| Q3 history model | Choices depend on past choice/reward | `IBL_behavior_dataQ3.ipynb` |
| Q4 sliding windows | Parameters can change within session | `IBL_behavior_dataQ4.ipynb` |
| Q5 reward/time model | Satiety or session progress affects choice | `IBL_behavior_dataQ5.ipynb` |
| Q8 GLM-HMM | Discrete hidden strategy states | `IBL_behavior_dataQ8_2.ipynb` |
| Q9 PsyTrack | Continuous latent weight trajectories | `IBL_behavior_dataQ9_3.ipynb` |
| Q10 final project | Sex-biased latent strategy dynamics | `IBL_behavior_sex_latent_strategy_project.ipynb` |

## Current caveats

- `ssm_new_full` supports the binary-choice GLM-HMM needed here, not the full original `ssm` library.
- `IBL_behavior_dataQ9_3.ipynb` requires `psytrack`; if the import fails, install it into the active Jupyter kernel.
- Some notebooks are intermediate copies preserved for development history. Use the recommended versions above for final results.
- Git status shows many notebooks as untracked or modified, so avoid deleting or overwriting files unless you intentionally want to clean the project.
