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

import numpy as np
from functions import *  # 导入激活函数和损失函数


# ============================================================================
# TODO 列表
# ============================================================================
# [ ] 1. 实现所有神经网络层的前向传播
# [ ] 2. 实现所有神经网络层的反向传播
# [ ] 3. 添加Dropout层
# [ ] 4. 添加BatchNormalization层
# [ ] 5. 添加卷积层
# [ ] 6. 添加池化层
# ============================================================================

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
        pass  # TODO: 实现ReLU前向传播

    def backward(self, dout):
        """
        反向传播

        参数:
            dout: numpy数组，上游梯度

        返回:
            numpy数组，对输入的梯度
        """
        pass  # TODO: 实现ReLU反向传播


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
        pass  # TODO: 实现Sigmoid前向传播

    def backward(self, dout):
        """
        反向传播

        参数:
            dout: numpy数组，上游梯度

        返回:
            numpy数组，对输入的梯度
        """
        pass  # TODO: 实现Sigmoid反向传播


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
        self.W = W
        self.b = b
        self.X = None
        self.X_original_shape = None
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        前向传播

        参数:
            X: numpy数组，输入数据

        返回:
            numpy数组，仿射变换后的输出
        """
        pass  # TODO: 实现仿射层前向传播

    def backward(self, dout):
        """
        反向传播

        参数:
            dout: numpy数组，上游梯度

        返回:
            numpy数组，对输入的梯度
        """
        pass  # TODO: 实现仿射层反向传播


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
        pass  # TODO: 实现SoftmaxWithLoss前向传播

    def backward(self, dout=1):
        """
        反向传播

        参数:
            dout: 标量，默认为1（损失对自身的导数为1）

        返回:
            numpy数组，对输入的梯度
        """
        pass  # TODO: 实现SoftmaxWithLoss反向传播


# ============================================================================
# 测试代码
# ============================================================================

def test_relu_layer():
    """测试ReLU层"""
    print("=" * 50)
    print("测试ReLU层")
    print("=" * 50)

    print("\n1. 测试前向传播:")
    test_input = np.array([[-1.0, 0.0, 1.0],
                           [-2.0, 2.0, -3.0]])
    print(f"输入形状: {test_input.shape}")
    print(f"输入值:\n{test_input}")
    print("期望输出:\n[[0., 0., 1.],\n [0., 2., 0.]]")

    print("\n2. 测试反向传播:")
    upstream_grad = np.array([[1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]])
    print(f"上游梯度:\n{upstream_grad}")
    print("期望输出梯度:\n[[0., 0., 1.],\n [0., 1., 0.]]")

    print("\n3. 测试mask属性:")
    print("前向传播后，mask应记录哪些位置 <= 0")
    print("mask应与输入形状相同，值为布尔类型")


def test_sigmoid_layer():
    """测试Sigmoid层"""
    print("\n" + "=" * 50)
    print("测试Sigmoid层")
    print("=" * 50)

    print("\n1. 测试前向传播:")
    test_input = np.array([-1.0, 0.0, 1.0])
    print(f"输入: {test_input}")
    print("期望输出: 所有值在(0, 1)范围内")
    print("具体值应接近: [0.2689, 0.5, 0.7311]")

    print("\n2. 测试y属性保存:")
    print("前向传播后，self.y应保存输出值")
    print("用于反向传播计算")

    print("\n3. 测试反向传播:")
    upstream_grad = np.array([0.1, 0.2, 0.3])
    print(f"上游梯度: {upstream_grad}")
    print("公式: dx = dout * y * (1 - y)")
    print("需要检查数值是否正确")


def test_affine_layer():
    """测试仿射层"""
    print("\n" + "=" * 50)
    print("测试仿射层")
    print("=" * 50)

    print("\n1. 测试初始化:")
    W = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
    b = np.array([0.1, 0.2])
    print(f"权重W形状: {W.shape}")
    print(f"偏置b形状: {b.shape}")

    print("\n2. 测试前向传播:")
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    print(f"输入X形状: {X.shape}")
    print(f"输入X:\n{X}")
    print("公式: Y = XW + b")
    print(f"期望Y形状: ({X.shape[0]}, {b.shape[0]})")

    print("\n3. 测试形状变换:")
    print("如果输入X不是二维，应展平为二维")
    X_3d = np.random.randn(2, 3, 4)  # 三维输入
    print(f"三维输入形状: {X_3d.shape}")
    print("期望: 展平为 (2, 12)")

    print("\n4. 测试反向传播:")
    print("需要测试:")
    print("a) 对输入的梯度dX")
    print("b) 对权重的梯度dW")
    print("c) 对偏置的梯度db")
    print("公式验证:")
    print("dX = dout · Wᵀ")
    print("dW = Xᵀ · dout")
    print("db = sum(dout, axis=0)")

    print("\n5. 测试梯度形状:")
    upstream_grad = np.ones((2, 2))
    print(f"上游梯度形状: {upstream_grad.shape}")
    print(f"期望dX形状: {X.shape}")
    print(f"期望dW形状: {W.shape}")
    print(f"期望db形状: {b.shape}")


def test_softmax_with_loss():
    """测试SoftmaxWithLoss层"""
    print("\n" + "=" * 50)
    print("测试SoftmaxWithLoss层")
    print("=" * 50)

    print("\n1. 测试one-hot编码标签:")
    x = np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0]])
    t_onehot = np.array([[0, 0, 1],
                         [0, 1, 0]])
    print(f"网络输出x:\n{x}")
    print(f"one-hot标签t:\n{t_onehot}")
    print("期望: 计算交叉熵损失")

    print("\n2. 测试类别索引标签:")
    t_indices = np.array([2, 1])  # 正确类别索引
    print(f"类别索引标签: {t_indices}")
    print("应与one-hot编码结果相同")

    print("\n3. 测试前向传播属性保存:")
    print("forward()后应保存:")
    print("self.y: Softmax输出")
    print("self.t: 目标标签")
    print("self.loss: 损失值")

    print("\n4. 测试反向传播:")
    print("one-hot编码情况: dx = (y - t) / batch_size")
    print("类别索引情况: dx = y.copy(); dx[n, t] -= 1; dx /= batch_size")

    print("\n5. 测试数值稳定性:")
    x_large = np.array([[1000.0, 1001.0, 1002.0]])
    print(f"大值输入: {x_large}")
    print("期望: 不应出现数值溢出")
    print("提示: 在softmax函数中实现数值稳定性处理")


def test_layer_chain():
    """测试层串联"""
    print("\n" + "=" * 50)
    print("测试层串联（小型神经网络）")
    print("=" * 50)

    print("\n构建网络: Affine -> ReLU -> Affine -> SoftmaxWithLoss")
    print("\n1. 定义各层:")
    print("   layer1: Affine(in_features=3, out_features=4)")
    print("   layer2: ReLU()")
    print("   layer3: Affine(in_features=4, out_features=2)")
    print("   layer4: SoftmaxWithLoss()")

    print("\n2. 前向传播流程:")
    print("   X -> layer1 -> Z1 -> layer2 -> A1 -> layer3 -> Z2 -> layer4 -> loss")

    print("\n3. 反向传播流程:")
    print("   loss <- layer4 <- layer3 <- layer2 <- layer1")
    print("   每层接收上游梯度，计算并传递下游梯度")

    print("\n4. 测试梯度流:")
    print("   确保梯度形状匹配")
    print("   确保梯度值合理（不全是0或NaN）")


def test_gradient_check():
    """测试梯度数值验证"""
    print("\n" + "=" * 50)
    print("测试梯度数值验证")
    print("=" * 50)

    print("\n1. 数值梯度 vs 解析梯度:")
    print("   使用数值微分计算梯度（作为参考）")
    print("   与反向传播计算的梯度比较")

    print("\n2. 测试Affine层梯度:")
    print("   对权重W和偏置b分别进行梯度检查")

    print("\n3. 测试ReLU层梯度:")
    print("   注意ReLU在x=0处的不可导性")

    print("\n4. 测试Sigmoid层梯度:")
    print("   检查梯度公式是否正确")

    print("\n5. 相对误差计算:")
    print("   error = |grad_analytic - grad_numeric| / max(|grad_analytic|, |grad_numeric|)")
    print("   期望: 相对误差 < 1e-7")


def run_all_tests():
    """运行所有测试"""
    print("神经网络层测试")
    print("=" * 50)

    try:
        test_relu_layer()
        test_sigmoid_layer()
        test_affine_layer()
        test_softmax_with_loss()
        test_layer_chain()
        test_gradient_check()

        print("\n" + "=" * 50)
        print("测试用例定义完成")
        print("请按顺序实现以下层:")
        print("1. Relu 层")
        print("2. Sigmoid 层")
        print("3. Affine 层")
        print("4. SoftmaxWithLoss 层")
        print("=" * 50)
        print("实现后运行测试验证正确性")
        print("特别注意梯度检查测试")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        print("可能是函数未正确实现或导入错误")


if __name__ == "__main__":
    run_all_tests()