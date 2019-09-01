from acc import acc
from make_graph import make_graph
from sklearn.cluster import SpectralClustering
import numpy as np
import time


def spectralclustering(X, y, n):
    """
    Spectral Clusteringによるクラスタリング

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
    acc_sc : float
        正解率
    time_sc : float
        実行時間
    """
    sc = SpectralClustering(n_clusters=2,
                            affinity="nearest_neighbors")
    start_sc = time.time()
    y_sc = sc.fit_predict(X)
    end_sc = time.time()
    y_sc = np.reshape(y_sc, (1, len(y[0])))
    acc_sc, _, _ = acc(y_sc, y)
    time_sc = round(end_sc - start_sc, 2)

    make_graph(X, y_sc, n, "SpectralClustering")

    return acc_sc, time_sc
