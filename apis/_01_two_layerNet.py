"""
    构建两层神经网络
    激活层第一层用sigmoid, 第二层用softmax
    用交叉熵得到损失值(y,t), 用数值微分进行计算(f,x),但最开始都只有x,t
"""
from commons.functions import ActivationFunctions, LossFunctions
import numpy as np
from commons.gradient import numerical_gradient


class TwoLayerNet:

    # 初始化
    def __init__(self, input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * weight_init_std,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * weight_init_std,
            'b2': np.zeros(output_size)
        }

    # 前向传播（预测）
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = x @ W1 + b1
        z1 = ActivationFunctions.sigmoid(a1)
        a2 = z1 @ W2 + b2
        y = ActivationFunctions.softmax(a2)
        return y

    # 计算准确度
    def accuracy(self, x, t):
        y = self.predict(x)
        y_label = np.argmax(y, axis=1)
        accuracy = np.sum(y_label == t) / y.shape[0]
        return accuracy

    # 损失函数
    def loss(self, x, t):
        y = self.predict(x)
        return LossFunctions.cross_entropy_error(y, t)

    # 计算梯度
    def numerical_gradient(self, x, t):
        loss_f = lambda w: self.loss(x, t)
        grads = {
            'W1': numerical_gradient(loss_f, self.params['W1']),
            'b1': numerical_gradient(loss_f, self.params['b1']),
            'W2': numerical_gradient(loss_f, self.params['W2']),
            'b2': numerical_gradient(loss_f, self.params['b2'])
        }
        return grads
