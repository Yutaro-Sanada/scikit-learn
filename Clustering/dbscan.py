from acc import acc
from make_graph import make_graph
from sklearn.cluster import DBSCAN
import numpy as np
import time


def dbscan(X, y, n):
    """
    DBSCANを用いたクラスタリング

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
    acc_dbs : float
        正解率
    time_dbs : float
        実行時間
    """
    dbs = DBSCAN(eps=0.1,
                 min_samples=2)
    start_dbs = time.time()
    y_dbs = dbs.fit_predict(X)
    end_dbs = time.time()
    y_dbs = np.reshape(y_dbs, (1, len(y[0])))
    acc_dbs, _, _ = acc(y_dbs, y)
    time_dbs = round(end_dbs - start_dbs, 2)

    make_graph(X, y_dbs, n, "DBSCAN")

    return acc_dbs, time_dbs
