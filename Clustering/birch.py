from acc import acc
from make_graph import make_graph
from sklearn.cluster import Birch
import numpy as np
import time


def birch(X, y, n):
    """
    Birchによるクラスタリング

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
    acc_br : float
        正解率
    time_br : float
        実行時間
    """
    br = Birch(n_clusters=2)
    start_br = time.time()
    y_br = br.fit_predict(X)
    end_br = time.time()
    y_br = np.reshape(y_br, (1, len(y[0])))
    acc_br, _, _ = acc(y_br, y)
    time_br = round(end_br - start_br, 2)

    make_graph(X, y_br, n, "Birch")

    return acc_br, time_br
