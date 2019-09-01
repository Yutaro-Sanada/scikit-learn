from acc import acc
from make_graph import make_graph
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import time


def agglomerativeclustering(X, y, n):
    """
    Agglomerative Clusteringを用いたクラスタリング

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
    acc_ac : float
        正解率
    time_ac : float
        実行時間
    -------
    """
    ac = AgglomerativeClustering(n_clusters=n)
    start_ac = time.time()
    y_ac = ac.fit_predict(X)
    end_ac = time.time()
    y_ac = np.reshape(y_ac, (1, len(y[0])))
    acc_ac, _, _ = acc(y_ac, y)
    time_ac = round(end_ac - start_ac, 2)

    make_graph(X, y_ac, n, "AgglomerativeClustering")

    return acc_ac, time_ac
