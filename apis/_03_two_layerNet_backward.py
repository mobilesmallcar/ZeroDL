# 两层神经网络用数值微分实现
"""
    构建反向传播神经网络
    1。两层神经网络
        全连接层用Affine,隐藏层用Relu,最后一层用softmaxWithLoss
    2. 构建顺序
        初始化
        前向传播/反向传播
        计算损失函数
        计算梯度/使用数值微分和反向传播,两种方法都写出来
        计算精准度
"""

import numpy as np
# from commons.functions import ActivationFunctions
from commons.layer import Affine, SoftmaxWithLoss, Relu
from collections import OrderedDict
from commons.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        # 初始化参数
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        # 定义层级结构
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.forward(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        accuracy = np.sum(y == t) / y.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        loss_f = lambda w: self.loss(x, t)
        grads = {
            'W1': numerical_gradient(loss_f, self.params['W1']),
            'b1': numerical_gradient(loss_f, self.params['b1']),
            'W2': numerical_gradient(loss_f, self.params['W2']),
            'b2': numerical_gradient(loss_f, self.params['b2'])
        }
        return grads

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())[::-1]
        for layer in layers:
            dout = layer.backward(dout)

        grads = {
            'W1': self.layers['Affine1'].dW,
            'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dW,
            'b2': self.layers['Affine2'].db
        }
        return grads
