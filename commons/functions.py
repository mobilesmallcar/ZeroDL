"""
神经网络激活函数与损失函数库
================================

本模块实现了神经网络中常用的激活函数和损失函数，包含数学公式说明和向量化实现。

数学符号说明：
- x: 输入值（标量、向量或矩阵）
- y: 输出值或预测值
- t: 目标值或真实标签
- np: numpy 模块
- exp(): 指数函数
- log(): 自然对数
- max(): 最大值函数
- sum(): 求和函数

作者:  xsy
版本: 1.0
"""

import numpy as np


# ============================================================================
# [ ] 1. 实现所有激活函数
# [ ] 2. 实现所有损失函数
# [ ] 3. 添加梯度计算函数
# [ ] 4. 添加性能测试
# [ ] 5. 添加文档示例
# ============================================================================

class ActivationFunctions:
    """激活函数类"""

    @staticmethod
    def step_function(x):
        """
        阶跃函数 (Step Function)

        数学公式:
            f(x) = 0  if x < 0
            f(x) = 1  if x >= 0

        参数:
            x: numpy数组，输入值

        返回:
            numpy数组，输出值（0或1）

        示例:
            # >>> step_function(np.array([-1, 0, 1]))
            array([0, 1, 1])
        """
        return np.where(x >= 0, 1, 0)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid 函数（逻辑函数）

        数学公式:
            f(x) = 1 / (1 + exp(-x))

        参数:
            x: numpy数组，输入值

        返回:
            numpy数组，输出值在(0, 1)范围内

        特性:
            - 输出范围: (0, 1)
            - 平滑可导
            - 常用于二分类问题的输出层
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        """
        ReLU 函数 (Rectified Linear Unit)

        数学公式:
            f(x) = max(0, x)

        参数:
            x: numpy数组，输入值

        返回:
            numpy数组，输出值

        特性:
            - 计算简单高效
            - 缓解梯度消失问题
            - 常用作隐藏层激活函数
        """
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        """
        Softmax 函数（归一化指数函数）

        数学公式:
            f(x_i) = exp(x_i) / Σ_j exp(x_j)

        参数:
            x: numpy数组，输入向量或矩阵
                - 如果是向量: 形状为 (C,)
                - 如果是矩阵: 形状为 (N, C)，N为样本数，C为类别数

        返回:
            numpy数组，概率分布（每个样本的类别概率和为1）

        特性:
            - 输出总和为1
            - 常用于多分类问题的输出层
            - 包含数值稳定性处理（防止指数溢出）
        """

        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def identity(x):
        """
        恒等函数（线性激活函数）

        数学公式:
            f(x) = x

        参数:
            x: numpy数组，输入值

        返回:
            numpy数组，输出值（与输入相同）

        特性:
            - 用于回归问题的输出层
            - 保持输入不变
        """
        return x


class LossFunctions:
    """损失函数类"""

    @staticmethod
    def mean_squared_error(y, t):
        """
        均方误差 (Mean Squared Error, MSE)

        数学公式:
            L = 0.5 * Σ(y_i - t_i)^2

        参数:
            y: numpy数组，预测值
            t: numpy数组，真实值

        返回:
            标量，损失值

        特性:
            - 常用于回归问题
            - 对离群点敏感
        """
        return 0.5 * np.sum((y - t) ** 2)

    @staticmethod
    def cross_entropy_error(y, t):
        """
        交叉熵误差 (Cross Entropy Error)

        数学公式:
            L = -Σ t_i * log(y_i)

        对于one-hot编码的标签:
            L = -log(y_k) 其中k是正确类别

        参数:
            y: numpy数组，预测概率分布
            t: numpy数组，真实标签（one-hot编码或类别索引）

        返回:
            标量，损失值

        特性:
            - 常用于分类问题
            - 包含数值稳定性处理（防止log(0)）
        """
        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)

        if y.size == t.size:
            t = t.argmax(axis=1)

        n = y.shape[0]
        return -np.sum(np.log(y[np.arange(n), t] + 1e-7)) / n


# ============================================================================
# 测试代码
# ============================================================================
def test_activation_functions():
    """测试激活函数"""
    print("=" * 50)
    print("测试激活函数")
    print("=" * 50)

    # 测试阶跃函数
    print("\n1. 测试阶跃函数:")
    test_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"输入: {test_input}")
    print(f"输出: {ActivationFunctions.step_function(test_input)}")
    print(f"期望输出: [0, 0, 1, 1, 1]")

    # 测试Sigmoid函数
    print("\n2. 测试Sigmoid函数:")
    test_input = np.array([-1.0, 0.0, 1.0])
    print(f"输入: {test_input}")
    print(f"期望输出范围: 所有值在(0, 1)之间")
    print(f"输出: {ActivationFunctions.sigmoid(test_input)}")
    print(f"期望对称性: sigmoid(-x) = 1 - sigmoid(x)")

    # 测试ReLU函数
    print("\n3. 测试ReLU函数:")
    test_input = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print(f"输入: {test_input}")
    print(f"期望输出: 所有值都大于等于0")
    print(f"输出: {ActivationFunctions.relu(test_input)}")
    print(f"期望输出: [0, 0, 0, 1, 3]")

    # 测试Softmax函数
    print("\n4. 测试Softmax函数:")
    test_input_vector = np.array([1.0, 2.0, 3.0])
    test_input_matrix = np.array([[1.0, 2.0, 3.0],
                                  [2.0, 4.0, 6.0]])
    print(f"向量输入: {test_input_vector}")
    print(f"矩阵输入形状: {test_input_matrix.shape}")
    print(f"输出: {ActivationFunctions.softmax(test_input_vector)}")
    print(f"输出矩阵: {ActivationFunctions.softmax(test_input_matrix)}")
    print(f"期望输出: 所有输出的和为1")
    print(f"期望特性: 较大输入值对应的输出概率更大")

    # 测试恒等函数
    print("\n5. 测试恒等函数:")
    test_input = np.array([1.0, 2.0, 3.0])
    print(f"输入: {test_input}")
    print(f"期望输出: {test_input} (应与输入完全相同)")


def test_loss_functions():
    """测试损失函数"""
    print("\n" + "=" * 50)
    print("测试损失函数")
    print("=" * 50)

    # 测试均方误差
    print("\n1. 测试均方误差:")
    y_pred = np.array([0.1, 0.2, 0.3])
    y_true = np.array([0.1, 0.2, 0.3])
    print(f"预测值: {y_pred}")
    print(f"真实值: {y_true}")
    print(f"输出: {LossFunctions.mean_squared_error(y_pred, y_true)}")
    print(f"期望输出: 0.0 (完美预测)")

    y_pred = np.array([0.1, 0.2, 0.3])
    y_true = np.array([0.0, 0.0, 0.0])
    print(f"\n预测值: {y_pred}")
    print(f"真实值: {y_true}")
    print(f"输出: {LossFunctions.mean_squared_error(y_pred, y_true)}")
    print(f"期望输出: 大于0的数值")

    # 测试交叉熵误差
    print("\n2. 测试交叉熵误差:")

    # 测试one-hot编码情况
    y_pred = np.array([[0.1, 0.8, 0.1],  # 预测概率
                       [0.3, 0.3, 0.4]])
    y_true_onehot = np.array([[0, 1, 0],  # 真实标签one-hot编码
                              [0, 0, 1]])
    print(f"预测概率:\n{y_pred}")
    print(f"真实标签(one-hot):\n{y_true_onehot}")
    print(f"输出: {LossFunctions.cross_entropy_error(y_pred, y_true_onehot)}")
    print("期望输出: 标量损失值")

    # 测试类别索引情况
    y_true_indices = np.array([1, 2])  # 正确类别索引
    print(f"\n真实标签(类别索引): {y_true_indices}")
    print(f"输出: {LossFunctions.cross_entropy_error(y_pred, y_true_indices)}")
    print("期望输出: 标量损失值")
    print("期望特性: 预测概率分布与真实标签越接近，损失越小")

    # 测试边界情况
    print("\n3. 测试边界情况:")
    y_pred_boundary = np.array([0.0, 0.0, 1.0])  # 有0值的情况
    y_true_boundary = np.array([0, 0, 1])
    print(f"预测值包含0: {y_pred_boundary}")
    print(f"真实值: {y_true_boundary}")
    print(f"输出: {LossFunctions.cross_entropy_error(y_pred_boundary, y_true_boundary)}")
    print("期望: 不应出现log(0)错误，应有数值稳定性处理")


def run_all_tests():
    """运行所有测试"""
    print("神经网络函数库测试开始")
    print("=" * 50)

    # try:
    test_activation_functions()
    test_loss_functions()

    print("\n" + "=" * 50)
    print("所有测试用例定义完成")
    print("请先实现各函数，然后运行测试验证正确性")
    print("=" * 50)

    # except Exception as e:
    #     print(f"\n测试过程中出现错误: {e}")
    #     print("请确保函数已正确实现")


if __name__ == "__main__":
    run_all_tests()
