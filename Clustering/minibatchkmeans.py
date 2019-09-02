from acc import acc
from make_graph import make_graph
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import time


def minibatchkmeans(X, y, n):
    """
    Mini Batch K Meansによるクラスタリング

    Parameters
    ----------
    X : numpy array
        データ
    y : numpy array
        正解ラベル
    n : int
        クラスタ数

    Returns
    -------
    acc_mbkm : float
        正解率
    time_mbkm : float
        実行時間
    """
    mbkm = MiniBatchKMeans(n_clusters=n,
                init="random",
                n_init=10,
                max_iter=300)
    start_mbkm = time.time()
    y_mbkm = mbkm.fit_predict(X)
    end_mbkm = time.time()
    y_mbkm = np.reshape(y_mbkm, (1, len(y[0])))
    acc_mbkm, _, _ = acc(y_mbkm, y)
    time_mbkm = round(end_mbkm - start_mbkm, 2)

    make_graph(X, y_mbkm, n, "MiniBatchKMeans")

    return acc_mbkm, time_mbkm
