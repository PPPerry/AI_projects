import matplotlib.pyplot as plt
import numpy as np


def load_planar_dataset():
    """
    生成二维花瓣形样本点
    :return: 样本坐标, 样本颜色
    """
    m = 400  # 样本数
    N = int(m / 2)  # 各类样本数
    D = 2  # 维度
    X = np.zeros((m, D))  # 400行2列，每一行是一个样本
    Y = np.zeros((m, 1), dtype='uint8')  # 标签向量，0代表红色，1代表蓝色

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = 10 * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def plot_decision_boundary(model, X, y):
    """
    绘制区分颜色区域的分界线
    :param model: 预测函数
    :param X: 样本坐标
    :param y: 样本颜色
    :return:
    """
    x_min = X[0, :].min() - 1
    x_max = X[0, :].max() + 1
    y_min = X[1, :].min() - 1
    y_max = X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model(np.c_[xx.ravel(), yy.ravel()])  # 预测整张画布上各个点的颜色
    Z = Z.reshape(xx.shape)  # 转换为(1, 400)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 这里仅需要找出分界线后填充，故使用等高线绘制，用时远小于使用散点图scatter绘制
    plt.ylabel('y')
    plt.xlabel('x')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def sigmoid(z):
    """
    sigmod函数实现
    :param z: 数值或一个numpy数组
    :return: [0, 1]范围数值
    """
    s = 1 / (1 + np.exp(-z))
    return s
