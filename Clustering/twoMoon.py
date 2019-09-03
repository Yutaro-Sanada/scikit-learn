from kmeans import kmeans
from agglomerativeclustering import agglomerativeclustering
from dbscan import dbscan
from spectralclustering import spectralclustering
from birch import birch
from minibatchkmeans import minibatchkmeans
from show_result import show_result
from make_graph import make_graph

from sklearn.utils import check_random_state, shuffle, check_array
from collections.abc import Iterable
import numpy as np
import numbers


def generator(n):
    """
    Parameter
    ---------
    n : int
        サンプル数

    Returns
    --------
    X : numpy array
        データ
    y : numpy array
        正解ラベル

    """
    n_samples = n  # サンプル数をここで与えてる
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    generator = check_random_state(100)

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])
    noise = None  # サンプルに与えるノイズはここで決める
    if shuffle:
        X, y = shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    make_graph(X, (y.astype('int32')).reshape(1,n), 2, "Original")
    return X, y

def main():
    n = 5000
    (X, y) = generator(n)
    X = X.astype('float32')
    y = y.astype('int32')
    y = y.reshape(1, n)

    show_result(kmeans(X, y, 2))
    show_result(spectralclustering(X, y, 2))
    show_result(agglomerativeclustering(X, y, 2))
    show_result(dbscan(X, y, 0.1, 2))
    show_result(birch(X, y, 2))
    show_result(minibatchkmeans(X, y, 2))

if __name__ == "__main__":
    main()
