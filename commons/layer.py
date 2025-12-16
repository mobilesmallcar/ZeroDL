"""
神经网络层实现库
================================

本模块实现了神经网络中的各种层（Layer），包括前向传播和反向传播功能。
这些层可以组合起来构建完整的神经网络。

数学符号说明：
- x: 输入数据
- y: 输出数据
- W: 权重矩阵
- b: 偏置向量
- dout: 上游梯度（反向传播的输入）
- dx: 反向传播的输出梯度
- dW: 权重的梯度
- db: 偏置的梯度
- t: 目标标签（真实值）

作者: xsy
版本: 1.0
"""


# ============================================================================
# [ ] 1. 实现所有神经网络层的前向传播
# [ ] 2. 实现所有神经网络层的反向传播
# [ ] 3. 添加Dropout层
# [ ] 4. 添加BatchNormalization层
# [ ] 5. 添加卷积层
# [ ] 6. 添加池化层
# ============================================================================
from .functions import ActivationFunctions,LossFunctions
import numpy as np

class Relu:
    """
    ReLU激活层

    前向传播:
        y = max(0, x)

    反向传播:
        dx = dout * (x > 0)

    特性:
        - 计算简单高效
        - 缓解梯度消失问题
        - 保存mask用于反向传播
    """

    def __init__(self):
        """
        初始化ReLU层

        属性:
            mask: 布尔数组，记录哪些输入小于等于0
        """
        self.mask = None

    def forward(self, x):
        """
        前向传播

        参数:
            x: numpy数组，输入数据

        返回:
            numpy数组，ReLU激活后的输出
        """
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    def backward(self, dout):
        """
        反向传播

        参数:
            dout: numpy数组，上游梯度

        返回:
            numpy数组，对输入的梯度
        """
        dx = dout.copy()
        dx[self.mask] = 0
        return dx


class Sigmoid:
    """
    Sigmoid激活层

    前向传播:
        y = 1 / (1 + exp(-x))

    反向传播:
        dx = dout * y * (1 - y)

    特性:
        - 输出范围(0, 1)
        - 平滑可导
        - 保存输出y用于反向传播
    """

    def __init__(self):
        """
        初始化Sigmoid层

        属性:
            y: 前向传播的输出，用于反向传播计算
        """
        self.y = None

    def forward(self, x):
        """
        前向传播

        参数:
            x: numpy数组，输入数据

        返回:
            numpy数组，Sigmoid激活后的输出
        """
        y = ActivationFunctions.sigmoid(x)
        self.y = y
        return y

    def backward(self, dout):
        """
        反向传播

        参数:
            dout: numpy数组，上游梯度

        返回:
            numpy数组，对输入的梯度
        """
        dx = dout * self.y * (1 - self.y)
        return dx


class Affine:
    """
    仿射变换层（全连接层）

    前向传播:
        Y = XW + b

    反向传播:
        dX = dout · Wᵀ
        dW = Xᵀ · dout
        db = sum(dout, axis=0)

    特性:
        - 包含权重W和偏置b
        - 保存输入X用于反向传播
        - 支持批量处理
    """

    def __init__(self, W, b):
        """
        初始化仿射层

        参数:
            W: numpy数组，权重矩阵
            b: numpy数组，偏置向量

        属性:
            W: 权重矩阵
            b: 偏置向量
            X: 输入数据
            X_original_shape: 输入数据的原始形状
            dW: 权重的梯度
            db: 偏置的梯度
        """
        self.W, self.b = W, b

        self.X = None
        self.X_original_shape = None

        # 参数的梯度,偏导数
        self.dW, self.db = None, None

    def forward(self, X):
        """
        前向传播

        参数:
            X: numpy数组，输入数据

        返回:
            numpy数组，仿射变换后的输出
        """
        self.X_original_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        Y = self.X @ self.W + self.b
        return Y

    def backward(self, dout):
        """
        反向传播

        参数:
            dout: numpy数组，上游梯度

        返回:
            numpy数组，对输入的梯度
        """
        dX = dout @ self.W.T
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dX.reshape(*self.X_original_shape)


class SoftmaxWithLoss:
    """
    Softmax + 交叉熵损失层

    前向传播:
        y = softmax(x)
        L = -Σ t_i * log(y_i)

    反向传播:
        dx = (y - t) / batch_size

    特性:
        - 组合了Softmax和交叉熵损失
        - 数值稳定性好
        - 支持两种标签格式：one-hot编码和类别索引
    """

    def __init__(self):
        """
        初始化SoftmaxWithLoss层

        属性:
            y: Softmax输出
            t: 目标标签
            loss: 损失值
        """
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        """
        前向传播（计算损失）

        参数:
            x: numpy数组，输入数据（网络输出）
            t: numpy数组，目标标签

        返回:
            标量，交叉熵损失值
        """
        self.t = t
        self.y = ActivationFunctions.softmax(x)
        self.loss = LossFunctions.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        反向传播

        参数:
            dout: 标量，默认为1（损失对自身的导数为1）

        返回:
            numpy数组，对输入的梯度
        """
        n = self.t.shape[0]

        if self.t.size == self.y.size:
            dx = self.y - self.t
        else:
            dx = self.y.copy()
            dx[np.arange(n), self.t] -= 1
        return dx / n
