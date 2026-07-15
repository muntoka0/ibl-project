# IBL behavior project

The project treats behavior as a sequence of increasingly rich models:

1. Fixed psychometric logistic regression.
2. Logistic regression with lapse probability.
3. Logistic regression with previous choice and previous reward history.
4. Sliding-window logistic regression for within-session nonstationarity.
5. Reward/time predictors for satiety or session-progress effects.


Use these as the current cleaned-up versions:

| File | Purpose | Status |
|---|---|---|
| `IBL_behavior_dataQ1.ipynb` | Baseline psychometric logistic regression across mice | Base analysis |
| `IBL_behavior_dataQ2.ipynb` | Logistic regression plus lapse probability | Base analysis |
| `IBL_behavior_dataQ3.ipynb` | Previous choice/reward history model | Base analysis |
| `IBL_behavior_dataQ4.ipynb` | Sliding-window stationarity analysis | Base analysis |
| `IBL_behavior_dataQ5.ipynb` | Reward fraction and trial fraction internal-state tests | Base analysis |


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


## Interpretation map

| Model | Internal-state assumption | Best file |
|---|---|---|
| Q1 logistic regression | One fixed psychometric strategy | `IBL_behavior_dataQ1.ipynb` |
| Q2 lapse model | Random lapses on top of fixed strategy | `IBL_behavior_dataQ2.ipynb` |
| Q3 history model | Choices depend on past choice/reward | `IBL_behavior_dataQ3.ipynb` |
| Q4 sliding windows | Parameters can change within session | `IBL_behavior_dataQ4.ipynb` |
| Q5 reward/time model | Satiety or session progress affects choice | `IBL_behavior_dataQ5.ipynb` |
