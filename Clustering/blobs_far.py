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
    n_samples = n
    generator = check_random_state(1)
    centers = None
    center_box = (-10.0, 10.0)
    n_features = 2
    cluster_std = 0.5
    centers = 2
    if isinstance(n_samples, numbers.Integral):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(center_box[0], center_box[1],
                                        size=(n_centers, n_features))

        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(center_box[0], center_box[1],
                                        size=(n_centers, n_features))
        try:
            assert len(centers) == n_centers
        except TypeError:
            raise ValueError("Parameter `centers` must be array-like. "
                             "Got {!r} instead".format(centers))
        except AssertionError:
            raise ValueError("Length of `n_samples` not consistent"
                             " with number of centers. Got n_samples = {} "
                             "and centers = {}".format(n_samples, centers))
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]

    # stds: if cluster_std is given as list, it must be consistent
    # with the n_centers
    if (hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers):
        raise ValueError("Length of `clusters_std` not consistent with "
                         "number of centers. Got centers = {} "
                         "and cluster_std = {}".format(centers, cluster_std))

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    X = []
    y = []

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(generator.normal(loc=centers[i], scale=std,
                                  size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)

    if shuffle:
        total_n_samples = np.sum(n_samples)
        indices = np.arange(total_n_samples)
        generator.shuffle(indices)
        X = X[indices]
        y = y[indices]
    make_graph(X, (y.astype('int32')).reshape(1, n*2), 2, "Original")
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
    show_result(dbscan(X, y, 1, 2))
    show_result(birch(X, y, 2))
    show_result(minibatchkmeans(X, y, 2))

if __name__ == "__main__":
    main()
