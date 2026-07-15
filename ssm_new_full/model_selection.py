import numpy as np


def cross_val_scores(model, datas, inputs=None, masks=None, tags=None, heldout_frac=0.1, n_repeats=3, **fit_kw):
    datas = datas if isinstance(datas, (list, tuple)) else [datas]
    inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs] * len(datas)
    scores = []
    rng = np.random.default_rng(0)

    for _ in range(n_repeats):
        repeat_scores = []
        for data, input_array in zip(datas, inputs):
            T = data.shape[0]
            idx = np.arange(T)
            rng.shuffle(idx)
            split = int((1 - heldout_frac) * T)
            train_idx = np.sort(idx[:split])
            test_idx = np.sort(idx[split:])
            model.fit(data[train_idx], inputs=input_array[train_idx], **fit_kw)
            repeat_scores.append(model.log_likelihood(data[test_idx], input=input_array[test_idx]) / max(len(test_idx), 1))
        scores.append(np.mean(repeat_scores))

    return np.asarray(scores)
