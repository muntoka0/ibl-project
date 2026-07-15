import numpy as np
from sklearn.decomposition import PCA


def pca_with_imputation(D, datas, masks=None, num_iters=20):
    datas = datas if isinstance(datas, (list, tuple)) else [datas]
    if masks is None:
        masks = [np.ones_like(data, dtype=bool) for data in datas]
    elif not isinstance(masks, (list, tuple)):
        masks = [masks]

    data = np.concatenate(datas)
    mask = np.concatenate(masks)
    full_data = data.copy()

    for col in range(full_data.shape[1]):
        good = mask[:, col]
        full_data[~good, col] = full_data[good, col].mean()

    for _ in range(num_iters):
        pca = PCA(D)
        x = pca.fit_transform(full_data)
        prediction = pca.inverse_transform(x)
        full_data[~mask] = prediction[~mask]

    ll = pca.score(full_data)
    xs = np.split(x, np.cumsum([len(data_item) for data_item in datas])[:-1])
    return pca, xs, ll


def interpolate_data(data, mask):
    data = np.asarray(data, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    for col in range(data.shape[1]):
        good = mask[:, col]
        if good.all():
            continue
        x = np.arange(data.shape[0])
        data[~good, col] = np.interp(x[~good], x[good], data[good, col])
    return data
