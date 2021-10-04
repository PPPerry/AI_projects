import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import sklearn
import sklearn.datasets


def sigmoid(Z):
    """
    sigmod函数实现
    :param Z: 数值或一个numpy数组
    :return: 输出A
    """
    A = 1 / (1 + np.exp(-Z))

    return A


def relu(Z):
    """
    relu函数实现
    :param Z: 数值或一个numpy数组
    :return: 输出A
    """
    A = np.maximum(0, Z)

    return A


def load_2D_dataset():
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()

    return train_X, train_Y, test_X, test_Y


def initialize_parameters(layer_dims):
    """
    初始化各层参数w和b
    :param layer_dims: 列表，存储每层的神经元个数
    :return: 参数
    """
    parameters = {}
    L = len(layer_dims)

    # 遍历每一层
    for l in range(1, L):
        # 若该层有3个神经元，上一层有2个神经元，则该层参数W的维度是3*2，b的维度是3*1
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_propagation(X, parameters):
    """
    前向传播
    :param X: 输入层样本
    :param parameters: 参数
    :return: 输出层的输出，中间变量
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)  # 第一层的激活函数使用relu
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)  # 第二层的激活函数使用relu
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)  # 第三层的激活函数用sigmoid

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    return a3, cache


def backward_propagation(X, Y, cache):
    """
    反向传播
    :param cache: 中间变量
    :param X: 样本坐标
    :param Y: 样本颜色
    :return: 梯度
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1. / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))  # relu的反向传播
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

    return gradients


def update_parameters(parameters, grads, learning_rate):
    """
    梯度下降
    :param parameters: 参数
    :param grads: 梯度
    :param learning_rate: 学习率
    :return: 更新后的参数
    """
    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters


def compute_loss(a3, Y):
    """
    计算成本
    :param a3: 输出层的输出
    :param Y: 样本颜色
    :return: 成本
    """
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3 + 1e-5), Y) + np.multiply(-np.log(1 - a3 + 1e-5), 1 - Y)
    loss = 1. / m * np.nansum(logprobs)

    return loss


def predict(X, y, parameters):
    """
    利用模型预测
    :param parameters: 参数
    :param X: 样本坐标
    :param y: 样本颜色
    :return: 预测的样本颜色
    """

    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    a3, caches = forward_propagation(X, parameters)

    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.mean((p[0, ] == y[0, :]))))

    return p


def plot_decision_boundary(model, X, y):
    """
    绘制区分颜色区域的分界线
    :param model: 预测函数
    :param X: 样本坐标
    :param y: 样本颜色
    :return:
    """
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model(np.c_[xx.ravel(), yy.ravel()])  # 预测整张画布上各个点的颜色
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 这里仅需要找出分界线后填充，故使用等高线绘制，用时远小于使用散点图scatter绘制
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3 > 0.5)

    return predictions

