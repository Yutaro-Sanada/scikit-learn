from acc import acc
from make_graph import make_graph
import numpy as np
from sklearn.cluster import KMeans
import time


def kmeans(X, y, n):
    """
    K Meansによるクラスタリング

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
    acc_km : float
        正解率
    time_km : float
        実行時間
    """
    km = KMeans(n_clusters=n,
                init="random",
                n_init=10,
                max_iter=300,
                random_state=0)
    start_km = time.time()
    y_km = km.fit_predict(X)
    end_km = time.time()
    y_km = np.reshape(y_km, (1, len(y[0])))
    acc_km, _, _ = acc(y_km, y)
    time_km = round(end_km - start_km, 2)

    make_graph(X, y_km, n, "KMeans")

    return acc_km, time_km
