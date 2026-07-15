
## `ssm_new_full` Summary

`ssm_new_full` is **not a full copy of original `ssm`**.  
It is a **Python 3.11-compatible rewrite of the GLM-HMM parts needed for Q8/Q9**.

### Fully Working For GLM-HMM

These files are implemented and used by the Q8/Q9 model:

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

They support:

```python
import ssm_new_full as ssm

model = ssm.HMM(
    K=3,
    D=1,
    M=3,
    observations="input_driven_obs",
    observation_kwargs={"C": 2}
)

model.fit(...)
model.expected_states(...)
model.most_likely_states(...)
model.transitions.transition_matrix
model.observations.weights
```

### Partially Ported Helpers

These files exist for compatibility and basic helper logic, but are **not full original `ssm` implementations**:

```text
stats.py
regression.py
model_selection.py
plots.py
preprocessing.py
primitives.py
hierarchical.py
```

They contain only the parts useful for this project or simple imports.

### Placeholders Only

These files are present so imports do not fail, but their full functionality is **not implemented**:

```text
lds.py
emissions.py
variational.py
extensions/
```

They require LDS/SLDS, Kalman smoothing, variational inference, autograd, and old compiled backend logic.

### Important Limits

This rewrite supports the project’s binary-choice GLM-HMM:

```text
D = 1
C = 2
observations = "input_driven_obs"
method = "em"
transitions = "standard" / "stationary"
choices shape = (T, 1)
choices = 0/1
```

### Bottom Line

`ssm_new_full` fully supports the **GLM-HMM chain** needed for Q8/Q9, but it is **not the full original `ssm` package**. It avoids the broken Python 3.10 `cstats/Cython` backend and works in Python 3.11.




If you mean **“what does this GLM-HMM support / not support?”**, then the honest answer is:

## GLM-HMM Supports

Your `ssm_new_full` supports this task:

```text
binary choice GLM-HMM
```

Meaning:

```text
hidden states + logistic regression choice model
```

It supports:

```python
model = ssm.HMM(
    K=3,
    D=1,
    M=3,
    observations="input_driven_obs",
    observation_kwargs={"C": 2}
)
```

Supported:

```text
choices = 0/1
choices shape = (T, 1)
inputs shape = (T, M)
D = 1
C = 2
method = "em"
observations = "input_driven_obs"
standard/stationary transitions
state inference with expected_states()
most likely states with most_likely_states()
state-specific GLM weights
transition matrix
```

So it supports your Q8/Q9 behavioral task.

---

## GLM-HMM Does Not Support

It does **not** support the full original `ssm` package.

Not supported:

```text
D > 1
C > 2
continuous observations
LDS
SLDS
Kalman smoothing
variational inference
hierarchical training
full emissions.py models
full autograd backend
compiled cstats backend
multinomial choices beyond binary 0/1
other observation types except input_driven_obs
```

Also not supported:

```python
observations="gaussian"
observations="bernoulli"
observations="categorical"
observations="ar"
transitions="sticky"
transitions="inputdriven"
```

unless you specifically implemented them.

---

Use this:

> This Python 3.11 rewrite supports the binary-choice input-driven GLM-HMM needed for Q8/Q9, but it does not support the full original `ssm` package, including LDS/SLDS, continuous emissions, variational inference, hierarchical models, or non-binary observation models.

Or shorter:

> It supports `ssm.HMM(..., observations="input_driven_obs")` for binary choices only. It does not support the full `ssm` library.