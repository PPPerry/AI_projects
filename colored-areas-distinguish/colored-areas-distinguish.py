import numpy as np
import matplotlib.pyplot as plt
import sklearn  # 用于数据挖掘、数据分析和机器学习
import sklearn.linear_model

from planar_utils import *  # 自定义工具函数

X, Y = load_planar_dataset()

# X[0, :]表示400点的横坐标，X[1, :]表示纵坐标，c=Y.ravel()是指定400个点的颜色，s=40指定点的大小
# cmap指定调色板，如果用不同的调色板，那么Y的值对应 的颜色也会不同，用plt.cm.Spectral这个调色板时，0指代红色，1指代蓝色
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.ylabel('y')
plt.xlabel('x')
plt.show()

# 生成单神经元网络并训练
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T.ravel())

single_predict = clf.predict(X.T)  # 对400个样本点的颜色预测结果
print('单神经元的预测准确率是：' + str(clf.score(X.T, Y.T.ravel()) * 100) + '%')

plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())


def initialize_parameters(n_x, n_h, n_y):
    """
    初始化参数w和b
    :param n_x: 输入层的神经元个数
    :param n_h: 隐藏层的神经元个数
    :param n_y: 输出层的神经元个数
    :return: 初始化的参数
    """
    # 随机初始化第一层（隐藏层）的参数，每一个隐藏层神经元都与输入层的每一个神经元相连
    W1 = np.random.randn(n_h, n_x) * 0.01  # W1的维度是(隐藏层的神经元个数, 输入层的神经元个数)
    b1 = np.zeros((n_h, 1))  # b1的维度是(隐藏层的神经元个数, 1)

    # 随机初始化第二层（输出层）的参数
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    """
    前向传播
    :param X: (2, 400)的输入层样本
    :param parameters: 参数
    :return: 输出层的输出，中间变量
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # 第一层的激活函数使用tanh
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # 第二层的激活函数用sigmoid

    # 保存中间变量，在反向传播时会用到
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    计算成本
    :param A2: 输出层的输出
    :param Y: 样本颜色
    :param parameters: 参数
    :return: 成本
    """
    m = Y.shape[1]

    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    反向传播
    :param parameters: 参数
    :param cache: 中间变量
    :param X: 样本坐标
    :param Y: 样本颜色
    :return: 梯度
    """
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.0):
    """
    梯度下降
    :param parameters: 参数
    :param grads: 梯度
    :param learning_rate: 学习率
    :return: 更新后的参数
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_module(X, Y, n_h, num_iterations=10001, print_cost=False):
    """
    构建神经网络模型
    :param X: 样本坐标
    :param Y: 样本颜色
    :param n_h: 隐藏层的神经元个数
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印成本
    :return: 参数
    """
    n_x = X.shape[0]  # 输入层的神经元个数
    n_y = Y.shape[0]  # 输出层的神经元个数

    # 初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        # 前向传播
        A2, cache = forward_propagation(X, parameters)

        # 计算成本
        cost = compute_cost(A2, Y, parameters)

        # 反向传播
        grads = backward_propagation(parameters, cache, X, Y)

        # 更新参数
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("在训练%i次后，成本是：%f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    利用模型预测
    :param parameters: 参数
    :param X: 样本坐标
    :return: 预测的样本颜色
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions


parameters = nn_module(X, Y, n_h=5, num_iterations=10001, print_cost=True)

predictions = predict(parameters, X)
print('浅层网络的预测准确率是：%d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# 绘制预测结果
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.ravel())

# 不同隐藏层的神经元个数对应的不同准确度
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    parameters = nn_module(X, Y, n_h, num_iterations=5000, print_cost=False)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print("{}个隐藏层神经元时的准确度是: {} %".format(n_h, accuracy))
