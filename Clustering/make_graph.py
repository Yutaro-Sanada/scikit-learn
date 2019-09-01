import matplotlib.pyplot as plt


def make_graph(X, y, n, method):
    """
    データを可視化する

    Parameters
    ----------
    X : numpy array
        データ
    y : numpy array
        正解ラベル
    n : int
        分類クラス数
    method : chr
        分類した方法
    Return
    ------
    None
    """
    print("可視化中")
    plt.figure(figsize=(8, 7))
    plt.title("Method : {0}, n_classes = {1}".format(method, n))
    colors = ['r', 'g', 'y']
    for i in range(len(y[0])):
        element = y[0][i]
        if (element == 0):
            plt.scatter(X[i][0], X[i][1], c=colors[0], marker='o', s=25)
        elif (element == 1):
            plt.scatter(X[i][0], X[i][1], c=colors[1], marker='o', s=25)
        else:
            plt.scatter(X[i][0], X[i][1], c=colors[2], marker='o', s=25)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    plt.close()
