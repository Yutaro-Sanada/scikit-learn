import numpy as np
from munkres import Munkres


def acc(y_predict, y_true):
    """
    精度を求める
    Parameters
    ----------
    y_predict : numpy array
        予測した結果
    y_true : numpy array
        正解ラベル
    Returns
    -------
    acc : int
        精度
    best_map : list
        偏りがないか確認
    y_pred : nmpy array
        予測した結果
    """
    y_pred = y_predict.copy()

    if len(np.unique(y_pred)) == len(np.unique(y_true)):
        C = len(np.unique(y_true))

        cost_m = np.zeros((C, C), dtype=float)
        for i in np.arange(0, C):
            a = np.where(y_pred == i)
            a = a[1]
            l = len(a)
            for j in np.arange(0, C):
                yj = np.ones((1, l)).reshape(1, l)
                yj = j * yj
                cost_m[i, j] = np.count_nonzero(yj - y_true[0, a])
        mk = Munkres()
        best_map = mk.compute(cost_m)

        (_, h) = y_pred.shape
        for i in np.arange(0, h):
            c = y_pred[0, i]
            v = best_map[c]
            v = v[1]
            y_pred[0, i] = v

        acc = 1 - (np.count_nonzero(y_pred - y_true) / h)

    else:
        acc = 0
    return acc, best_map, y_pred
