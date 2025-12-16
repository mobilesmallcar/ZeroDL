"""
优化算法库
================================

本模块实现了神经网络训练中常用的优化算法，用于更新模型参数。
包括基本的随机梯度下降法及其多种变体。

数学符号说明：
- θ: 模型参数
- η: 学习率 (lr)
- ∇L(θ): 损失函数关于参数的梯度
- v: 动量项（速度）
- h: 梯度平方的累积项
- α: 动量系数或衰减率
- t: 时间步（迭代次数）
- ε: 小常数，防止除零（通常取1e-7）

作者: 优化算法实现者
版本: 1.0
"""

import numpy as np


# ============================================================================
# TODO 列表
# ============================================================================
# [ ] 1. 实现所有优化算法
# [ ] 2. 添加学习率调度器
# [ ] 3. 添加梯度裁剪功能
# [ ] 4. 添加权重衰减（L2正则化）
# [ ] 5. 添加Nesterov动量
# [ ] 6. 添加AdamW优化器
# ============================================================================

class SGD:
    """
    随机梯度下降法 (Stochastic Gradient Descent)

    更新公式:
        θ_{t+1} = θ_t - η * ∇L(θ_t)

    特性:
        - 最简单的优化算法
        - 固定学习率
        - 可能收敛较慢或在峡谷中震荡
    """

    def __init__(self, lr=0.01):
        """
        初始化SGD优化器

        参数:
            lr: 学习率 (η)
        """
        self.lr = lr

    def update(self, params, grads):
        """
        更新参数

        参数:
            params: 字典，包含所有参数 {key: parameter_array}
            grads: 字典，包含所有梯度 {key: gradient_array}

        返回:
            无，直接修改params
        """
        pass  # TODO: 实现SGD参数更新


class Momentum:
    """
    动量法 (Momentum)

    更新公式:
        v_{t+1} = α * v_t - η * ∇L(θ_t)
        θ_{t+1} = θ_t + v_{t+1}

    特性:
        - 引入动量项，加速收敛
        - 减少震荡
        - α通常取0.9
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        初始化Momentum优化器

        参数:
            lr: 学习率 (η)
            momentum: 动量系数 (α)

        属性:
            v: 动量项（速度）
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """
        更新参数

        参数:
            params: 字典，包含所有参数
            grads: 字典，包含所有梯度

        返回:
            无，直接修改params
        """
        pass  # TODO: 实现Momentum参数更新


class AdaGrad:
    """
    AdaGrad (Adaptive Gradient Algorithm)

    更新公式:
        h_{t+1} = h_t + ∇L(θ_t) ⊙ ∇L(θ_t)
        θ_{t+1} = θ_t - η * ∇L(θ_t) / (√h_{t+1} + ε)

    特性:
        - 自适应学习率
        - 为每个参数调整学习率
        - 适合稀疏数据
        - 学习率可能过早衰减
    """

    def __init__(self, lr=0.01):
        """
        初始化AdaGrad优化器

        参数:
            lr: 初始学习率

        属性:
            h: 梯度平方的累积
        """
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        """
        更新参数

        参数:
            params: 字典，包含所有参数
            grads: 字典，包含所有梯度

        返回:
            无，直接修改params
        """
        pass  # TODO: 实现AdaGrad参数更新


class RMSProp:
    """
    RMSProp (Root Mean Square Propagation)

    更新公式:
        h_{t+1} = α * h_t + (1 - α) * ∇L(θ_t) ⊙ ∇L(θ_t)
        θ_{t+1} = θ_t - η * ∇L(θ_t) / (√h_{t+1} + ε)

    特性:
        - 改进的AdaGrad，解决学习率衰减问题
        - 使用指数移动平均
        - α通常取0.9
    """

    def __init__(self, lr=0.01, alpha=0.9):
        """
        初始化RMSProp优化器

        参数:
            lr: 学习率
            alpha: 衰减率

        属性:
            h: 梯度平方的指数移动平均
        """
        self.lr = lr
        self.alpha = alpha
        self.h = None

    def update(self, params, grads):
        """
        更新参数

        参数:
            params: 字典，包含所有参数
            grads: 字典，包含所有梯度

        返回:
            无，直接修改params
        """
        pass  # TODO: 实现RMSProp参数更新


class Adam:
    """
    Adam (Adaptive Moment Estimation)

    更新公式:
        m_{t+1} = β₁ * m_t + (1 - β₁) * ∇L(θ_t)           # 一阶矩估计
        v_{t+1} = β₂ * v_t + (1 - β₂) * ∇L(θ_t) ⊙ ∇L(θ_t)  # 二阶矩估计
        m̂_t = m_t / (1 - β₁^t)                           # 偏差修正
        v̂_t = v_t / (1 - β₂^t)
        θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)

    特性:
        - 结合了动量和自适应学习率
        - 默认参数通常效果很好
        - 需要偏差修正
    """

    def __init__(self, lr=0.01, alpha1=0.9, alpha2=0.999):
        """
        初始化Adam优化器

        参数:
            lr: 学习率
            alpha1: 一阶矩的衰减率 (β₁)
            alpha2: 二阶矩的衰减率 (β₂)

        属性:
            v: 一阶矩（动量）
            h: 二阶矩（梯度平方）
            t: 时间步（迭代次数）
        """
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.v = None
        self.h = None
        self.t = 0

    def update(self, params, grads):
        """
        更新参数

        参数:
            params: 字典，包含所有参数
            grads: 字典，包含所有梯度

        返回:
            无，直接修改params
        """
        pass  # TODO: 实现Adam参数更新


# ============================================================================
# 测试代码
# ============================================================================

def test_optimizer_initialization():
    """测试优化器初始化"""
    print("=" * 50)
    print("测试优化器初始化")
    print("=" * 50)

    print("\n1. 测试SGD初始化:")
    print("   sgd = SGD(lr=0.01)")
    print("   期望: lr = 0.01")

    print("\n2. 测试Momentum初始化:")
    print("   momentum = Momentum(lr=0.01, momentum=0.9)")
    print("   期望: lr = 0.01, momentum = 0.9, v = None")

    print("\n3. 测试AdaGrad初始化:")
    print("   adagrad = AdaGrad(lr=0.01)")
    print("   期望: lr = 0.01, h = None")

    print("\n4. 测试RMSProp初始化:")
    print("   rmsprop = RMSProp(lr=0.01, alpha=0.9)")
    print("   期望: lr = 0.01, alpha = 0.9, h = None")

    print("\n5. 测试Adam初始化:")
    print("   adam = Adam(lr=0.01, alpha1=0.9, alpha2=0.999)")
    print("   期望: lr = 0.01, alpha1 = 0.9, alpha2 = 0.999")
    print("         v = None, h = None, t = 0")


def test_sgd_optimizer():
    """测试SGD优化器"""
    print("\n" + "=" * 50)
    print("测试SGD优化器")
    print("=" * 50)

    print("\n1. 基本参数更新:")
    params = {'W1': np.array([[1.0, 2.0], [3.0, 4.0]]),
              'b1': np.array([0.1, 0.2])}
    grads = {'W1': np.array([[0.1, 0.2], [0.3, 0.4]]),
             'b1': np.array([0.01, 0.02])}
    print(f"初始参数W1:\n{params['W1']}")
    print(f"梯度W1:\n{grads['W1']}")
    print(f"学习率: 0.01")
    print("期望: W1 -= 0.01 * grads['W1']")

    print("\n2. 测试多个参数:")
    print("   确保所有参数都被正确更新")

    print("\n3. 测试不同学习率:")
    print("   lr=0.1时，更新幅度应更大")
    print("   lr=0.001时，更新幅度应更小")


def test_momentum_optimizer():
    """测试Momentum优化器"""
    print("\n" + "=" * 50)
    print("测试Momentum优化器")
    print("=" * 50)

    print("\n1. 第一次更新:")
    print("   初始化时v应为None")
    print("   第一次update应初始化v为零矩阵")
    print("   更新公式: v = -lr * grads, params += v")

    print("\n2. 第二次更新:")
    print("   动量项应起作用")
    print("   更新公式: v = momentum * v - lr * grads")
    print("           params += v")

    print("\n3. 测试动量效应:")
    print("   如果梯度方向相同，更新速度应加快")
    print("   如果梯度方向变化，动量可减少震荡")


def test_adagrad_optimizer():
    """测试AdaGrad优化器"""
    print("\n" + "=" * 50)
    print("测试AdaGrad优化器")
    print("=" * 50)

    print("\n1. 第一次更新:")
    print("   初始化时h应为None")
    print("   第一次update应初始化h为零矩阵")
    print("   h += grads * grads")
    print("   params -= lr * grads / (sqrt(h) + 1e-7)")

    print("\n2. 自适应学习率特性:")
    print("   对于频繁更新的参数，h较大，学习率较小")
    print("   对于不频繁更新的参数，h较小，学习率较大")

    print("\n3. 测试数值稳定性:")
    print("   分母加1e-7防止除零")
    print("   确保sqrt(h)不为负")


def test_rmsprop_optimizer():
    """测试RMSProp优化器"""
    print("\n" + "=" * 50)
    print("测试RMSProp优化器")
    print("=" * 50)

    print("\n1. 更新公式:")
    print("   h = alpha * h + (1 - alpha) * grads * grads")
    print("   params -= lr * grads / (sqrt(h) + 1e-7)")

    print("\n2. 与AdaGrad比较:")
    print("   RMSProp使用指数移动平均")
    print("   不会像AdaGrad那样过度降低学习率")

    print("\n3. 测试衰减率alpha:")
    print("   alpha=0.9: 历史信息权重较大")
    print("   alpha=0.5: 更关注近期梯度")


def test_adam_optimizer():
    """测试Adam优化器"""
    print("\n" + "=" * 50)
    print("测试Adam优化器")
    print("=" * 50)

    print("\n1. 初始化状态:")
    print("   v: 一阶矩（类似动量）")
    print("   h: 二阶矩（类似RMSProp）")
    print("   t: 时间步，从0开始")

    print("\n2. 第一次更新:")
    print("   t = 1")
    print("   初始化v和h为零矩阵")
    print("   更新v和h（带偏差修正）")

    print("\n3. 偏差修正公式:")
    print("   lr_t = lr * sqrt(1 - alpha2^t) / (1 - alpha1^t)")
    print("   修正早期迭代的偏差")

    print("\n4. 更新步骤:")
    print("   v += (1 - alpha1) * (grads - v)")
    print("   h += (1 - alpha2) * (grads^2 - h)")
    print("   params -= lr_t * v / (sqrt(h) + 1e-7)")

    print("\n5. 测试时间步更新:")
    print("   每次update()时t应增加1")


def test_optimizer_comparison():
    """测试优化器比较"""
    print("\n" + "=" * 50)
    print("测试优化器比较")
    print("=" * 50)

    print("\n1. 相同初始条件和梯度:")
    print("   使用相同的params和grads")
    print("   比较不同优化器的更新结果")

    print("\n2. 测试场景:")
    print("   a) 简单二次函数优化")
    print("   b) 峡谷地形（一个方向梯度大，一个方向梯度小）")
    print("   c) 稀疏梯度场景")

    print("\n3. 期望特性:")
    print("   SGD: 简单直接，可能震荡")
    print("   Momentum: 减少震荡，加速收敛")
    print("   AdaGrad: 自适应学习率，适合稀疏数据")
    print("   RMSProp: 改进的AdaGrad")
    print("   Adam: 结合动量和自适应学习率")


def test_convergence_simulation():
    """测试收敛模拟"""
    print("\n" + "=" * 50)
    print("测试收敛模拟（简单示例）")
    print("=" * 50)

    print("\n优化函数: f(x) = x^2")
    print("最小值点: x = 0")

    print("\n1. 初始化:")
    params = {'x': np.array([10.0])}  # 初始点 x=10
    print(f"初始参数: x = {params['x'][0]}")

    print("\n2. 梯度计算:")
    print("   f'(x) = 2x")
    print("   在x=10处，梯度为20")

    print("\n3. 不同优化器行为预测:")
    print("   SGD: x_{t+1} = x_t - 0.01 * 2x_t")
    print("   Momentum: 有动量积累")
    print("   AdaGrad: 学习率逐渐减小")
    print("   Adam: 结合动量和自适应学习率")


def run_all_tests():
    """运行所有测试"""
    print("优化算法库测试")
    print("=" * 50)

    try:
        test_optimizer_initialization()
        test_sgd_optimizer()
        test_momentum_optimizer()
        test_adagrad_optimizer()
        test_rmsprop_optimizer()
        test_adam_optimizer()
        test_optimizer_comparison()
        test_convergence_simulation()

        print("\n" + "=" * 50)
        print("测试用例定义完成")
        print("请按顺序实现以下优化器:")
        print("1. SGD (最简单，先实现)")
        print("2. Momentum")
        print("3. AdaGrad")
        print("4. RMSProp")
        print("5. Adam (最复杂，最后实现)")
        print("=" * 50)
        print("实现后，可以运行测试验证:")
        print("1. 参数是否正确更新")
        print("2. 学习率是否起作用")
        print("3. 动量/自适应特性是否工作")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        print("可能是优化器未正确实现")


if __name__ == "__main__":
    run_all_tests()