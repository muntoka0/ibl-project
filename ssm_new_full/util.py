import numpy as np
from scipy.optimize import linear_sum_assignment

SEED = hash("ssm_new_full") % (2**32)
LOG_EPS = 1e-12
DIV_EPS = 1e-12


def logsumexp(values, axis=None, keepdims=False):
    values = np.asarray(values, dtype=float)
    max_value = np.max(values, axis=axis, keepdims=True)
    max_value = np.where(np.isfinite(max_value), max_value, 0.0)
    summed = np.sum(np.exp(values - max_value), axis=axis, keepdims=True)
    output = max_value + np.log(summed + LOG_EPS)

    if not keepdims:
        output = np.squeeze(output, axis=axis)

    return output


def sigmoid(values):
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def normalize(probabilities, axis=-1):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 0.0, np.inf)
    total = probabilities.sum(axis=axis, keepdims=True)
    total = np.where(total <= 0, 1.0, total)
    return probabilities / total


def ensure_data(data):
    data = np.asarray(data)

    if data.ndim == 1:
        data = data[:, None]

    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError("choices/data must have shape (T, 1) or (T,).")

    return data.astype(int)


def ensure_input(input_array, T, M):
    if input_array is None:
        if M == 0:
            return np.zeros((T, 0))
        raise ValueError("inputs/input is required when M > 0.")

    input_array = np.asarray(input_array, dtype=float)

    if input_array.ndim == 1:
        input_array = input_array[:, None]

    if input_array.ndim != 2:
        raise ValueError("inputs/input must be a 2D array with shape (T, M).")

    if input_array.shape[0] != T:
        raise ValueError("inputs/input must have the same number of rows as choices.")

    if input_array.shape[1] != M:
        raise ValueError(f"expected inputs/input with M={M} columns, got {input_array.shape[1]}.")

    return input_array


def ensure_list_data_and_inputs(data, inputs, M):
    if isinstance(data, (list, tuple)):
        datas = [ensure_data(item) for item in data]
        if inputs is None:
            input_list = [None] * len(datas)
        elif isinstance(inputs, (list, tuple)):
            input_list = list(inputs)
        else:
            raise ValueError("when data is a list, inputs must also be a list or None.")
    else:
        datas = [ensure_data(data)]
        input_list = [inputs]

    if len(datas) != len(input_list):
        raise ValueError("data and inputs must contain the same number of sequences.")

    input_list = [
        ensure_input(input_item, data_item.shape[0], M)
        for data_item, input_item in zip(datas, input_list)
    ]

    return datas, input_list


def logistic(x):
    return sigmoid(x)


def relu(x):
    return np.maximum(x, 0)


def logit(p):
    p = np.clip(p, LOG_EPS, 1 - LOG_EPS)
    return np.log(p / (1 - p))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def inv_softplus(y):
    y = np.asarray(y)
    return y + np.log1p(-np.exp(-y))


def one_hot(x, K):
    x = np.asarray(x, dtype=int).reshape(-1)
    out = np.zeros((len(x), K))
    out[np.arange(len(x)), x] = 1
    return out


def rle(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    change_points = np.concatenate(([0], np.where(x[1:] != x[:-1])[0] + 1, [len(x)]))
    starts = change_points[:-1]
    ends = change_points[1:]
    values = x[starts]
    lengths = ends - starts
    return values, lengths, starts


def check_shape(name, value, expected_shape):
    if np.shape(value) != tuple(expected_shape):
        raise ValueError(f"{name} has shape {np.shape(value)}, expected {expected_shape}.")


def compute_state_overlap(z1, z2, K1=None, K2=None):
    z1 = np.asarray(z1, dtype=int)
    z2 = np.asarray(z2, dtype=int)
    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2
    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    _, perm = linear_sum_assignment(-overlap)
    return perm


def trace_product(A, B):
    return np.sum(A * np.swapaxes(B, -1, -2))


def ensure_args_are_lists(fn):
    def wrapper(self, datas, inputs=None, masks=None, tags=None, *args, **kwargs):
        datas = datas if isinstance(datas, (list, tuple)) else [datas]
        inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs] * len(datas)
        masks = masks if isinstance(masks, (list, tuple)) else [masks] * len(datas)
        tags = tags if isinstance(tags, (list, tuple)) else [tags] * len(datas)
        return fn(self, datas, inputs=inputs, masks=masks, tags=tags, *args, **kwargs)
    return wrapper


ensure_args_not_none = ensure_args_are_lists
ensure_slds_args_not_none = ensure_args_are_lists
ensure_variational_args_are_lists = ensure_args_are_lists


def replicate(x, n):
    return [x for _ in range(n)]


def collapse(xs):
    return xs[0] if isinstance(xs, (list, tuple)) and len(xs) == 1 else xs


def ssm_pbar(num_iters, verbose=2, description=""):
    try:
        from tqdm.auto import trange
        return trange(num_iters, disable=verbose == 0, desc=description)
    except Exception:
        return range(num_iters)
