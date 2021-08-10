import numpy as np


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


def sigmoid_backward(dA, cache):
    """
    sigmoid函数的反向传播
    :param dA: 这一层的dA
    :param cache: Z
    :return: dZ
    """
    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ


def relu_backward(dA, cache):
    """
    relu函数的反向传播
    :param dA: 这一层的dA
    :param cache: Z
    :return: dZ
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # dZ = dA

    dZ[Z <= 0] = 0  # z小于等于0时，dZ = 0

    return dZ
