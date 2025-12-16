"""
数值微分与梯度计算库
================================

本模块实现了数值微分和数值梯度计算函数，用于近似计算函数的导数和梯度。
采用中心差分法提高计算精度。

数学符号说明：
- f: 函数（标量函数或多元函数）
- x: 自变量（标量、向量或矩阵）
- h: 微小量（通常取1e-4）
- ∂f/∂x: 函数f对x的偏导数
- ∇f: 函数f的梯度向量
- np: numpy模块

作者:  xsy
版本: 1.0
"""

import numpy as np


# ============================================================================
# TODO 列表
# ============================================================================
# [ ] 1. 实现数值微分函数
# [ ] 2. 实现数值梯度计算函数
# [ ] 3. 添加向量化梯度计算
# [ ] 4. 添加性能优化（并行计算）
# [ ] 5. 添加高阶导数计算
# ============================================================================

def numerical_diff0(f, x):
    """
    前向差分数值微分

    数学公式:
        f'(x) ≈ (f(x+h) - f(x)) / h

    参数:
        f: 函数，输入标量返回标量
        x: 标量，自变量值

    返回:
        标量，函数在x处的近似导数值

    注意:
        - 使用前向差分，精度较低
        - h固定为1e-4
        - 适用于标量函数
    """
    h = 1e-4
    return (f(x + h) - f(x)) / h


def numerical_diff(f, x):
    """
    中心差分数值微分（精度更高）

    数学公式:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    参数:
        f: 函数，输入标量返回标量
        x: 标量，自变量值

    返回:
        标量，函数在x处的近似导数值

    特性:
        - 使用中心差分，精度更高
        - 截断误差为O(h²)
        - h固定为1e-4
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def _numerical_gradient(f, x):
    """
    计算多元函数的梯度向量（内部函数）

    数学公式:
        ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ
        ∂f/∂xᵢ ≈ (f(x₁,...,xᵢ+h,...,xₙ) - f(x₁,...,xᵢ-h,...,xₙ)) / (2h)

    参数:
        f: 多元函数，输入向量返回标量
        x: numpy一维数组，自变量向量 [x₁, x₂, ..., xₙ]

    返回:
        numpy一维数组，梯度向量

    注意:
        - 使用中心差分法
        - 逐个计算每个方向的偏导数
        - 保持x的原始值不变
    """
    grad = np.zeros_like(x)
    h = 1e-4

    for i in range(x.size):
        temp = x[i]
        x[i] = temp + h
        fxh1 = f(x)
        x[i] = temp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = temp
    return grad


def numerical_gradient(f, X):
    """
    数值梯度计算（支持向量和矩阵输入）

    对于一维输入X:
        返回梯度向量 ∇f(X)

    对于二维输入X (m×n 矩阵):
        对每一行计算梯度，返回 m×n 梯度矩阵

    参数:
        f: 函数
            - 如果X是一维: f接受一维数组返回标量
            - 如果X是二维: f接受一维数组返回标量
        X: numpy数组
            - 一维: [x₁, x₂, ..., xₙ]
            - 二维: [[x₁₁, x₁₂, ...], [x₂₁, x₂₂, ...], ...]

    返回:
        numpy数组，梯度

    示例:
        # >>> def f(x): return x[0]**2 + x[1]**2
        # >>> numerical_gradient(f, np.array([3.0, 4.0]))
        array([6., 8.])
    """
    if X.ndim == 1:
        return _numerical_gradient(f, X)

    grad = np.zeros_like(X)
    for i, x in enumerate(X):
        grad[i] = _numerical_gradient(f, x)
    return grad


# ============================================================================
# 辅助函数（可选实现）
# ============================================================================

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降法

    数学公式:
        x_{k+1} = x_k - η * ∇f(x_k)

    参数:
        f: 目标函数
        init_x: 初始点
        lr: 学习率 (η)
        step_num: 迭代次数

    返回:
        最终的点x和过程历史

    注意:
        - 需要先实现numerical_gradient函数
    """
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())   # 保存当前点到列表
        grad = numerical_gradient(f, x)   # 计算梯度
        x = x - lr * grad    # 更新点
    return x, np.array(x_history)


# ============================================================================
# 测试代码
# ============================================================================

def test_numerical_diff():
    """测试数值微分函数"""
    print("=" * 50)
    print("测试数值微分")
    print("=" * 50)

    # 测试1: 线性函数 f(x) = 2x
    print("\n1. 测试线性函数 f(x) = 2x:")

    def linear_func(x):
        return 2 * x

    test_x = 3.0
    print(f"在 x = {test_x} 处的导数")
    print(f"理论值: 2.0")
    print("期望: numerical_diff0 和 numerical_diff 都应接近 2.0")
    print("注意: numerical_diff 应该更精确")

    # 测试2: 二次函数 f(x) = x^2
    print("\n2. 测试二次函数 f(x) = x^2:")

    def quadratic_func(x):
        return x ** 2

    test_x = 2.0
    print(f"在 x = {test_x} 处的导数")
    print(f"理论值: 4.0")
    print("期望误差应小于 1e-7")

    # 测试3: 指数函数 f(x) = exp(x)
    print("\n3. 测试指数函数 f(x) = exp(x):")

    def exp_func(x):
        return np.exp(x)

    test_x = 1.0
    print(f"在 x = {test_x} 处的导数")
    print(f"理论值: {np.exp(1.0):.6f}")

    # 测试4: 正弦函数 f(x) = sin(x)
    print("\n4. 测试正弦函数 f(x) = sin(x):")

    def sin_func(x):
        return np.sin(x)

    test_x = np.pi / 4  # 45度
    print(f"在 x = π/4 处的导数")
    print(f"理论值: cos(π/4) = {np.cos(np.pi / 4):.6f}")

    # 测试5: 边界情况
    print("\n5. 测试边界情况:")
    print("测试 x = 0 的情况")
    print("测试负值 x = -2.0")
    print("测试大值 x = 100.0")


def test_numerical_gradient():
    """测试数值梯度计算"""
    print("\n" + "=" * 50)
    print("测试数值梯度")
    print("=" * 50)

    # 测试1: 二元二次函数 f(x,y) = x^2 + y^2
    print("\n1. 测试二元二次函数 f(x,y) = x^2 + y^2:")

    def func2d(x):
        return x[0] ** 2 + x[1] ** 2

    test_point = np.array([3.0, 4.0])
    print(f"在点 {test_point} 处的梯度")
    print(f"数值梯度: [2x, 2y] = {numerical_gradient(func2d, test_point)}")
    print(f"理论梯度: [2x, 2y] = [6.0, 8.0]")

    # 测试2: 三元线性函数 f(x,y,z) = 2x + 3y + 4z
    print("\n2. 测试三元线性函数 f(x,y,z) = 2x + 3y + 4z:")

    def func3d(x):
        return 2 * x[0] + 3 * x[1] + 4 * x[2]

    test_point = np.array([1.0, 2.0, 3.0])
    print(f"在点 {test_point} 处的梯度")
    print(f"数值梯度: [2x, 3y, 4z] = {numerical_gradient(func3d, test_point)}")
    print(f"理论梯度: [2x, 3y, 4z] = [2.0, 3.0, 4.0] (常数梯度)")

    # 测试3: 高维函数（5维）
    print("\n3. 测试高维函数 (5维):")

    def func5d(x):
        return np.sum(x ** 2)  # 球面函数

    test_point = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"在点 {test_point} 处的梯度")
    print(f"数值梯度: {numerical_gradient(func5d, test_point)}")
    print(f"理论梯度: [2, 4, 6, 8, 10]")

    # 测试4: 矩阵输入（批量计算）
    print("\n4. 测试矩阵输入（批量梯度计算）:")
    test_points = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
    print(f"输入矩阵形状: {test_points.shape}")
    print(f"期望输出矩阵形状: {test_points.shape}")
    print("\n理论梯度（对于 f(x,y)=x²+y²，梯度为[2x, 2y]）:")
    print(f"点 [1.0, 2.0]: 理论梯度 = [2*1.0, 2*2.0] = [2.0, 4.0]")
    print(f"点 [3.0, 4.0]: 理论梯度 = [2*3.0, 2*4.0] = [6.0, 8.0]")
    print(f"点 [5.0, 6.0]: 理论梯度 = [2*5.0, 2*6.0] = [10.0, 12.0]")
    print(f"输出: {numerical_gradient(func5d, test_points)}")

    # 测试5: 复杂函数
    print("\n5. 测试复杂函数 f(x,y) = sin(x) * cos(y):")

    def complex_func(x):
        return np.sin(x[0]) * np.cos(x[1])

    test_point = np.array([np.pi / 4, np.pi / 3])
    print(f"在点 [π/4, π/3] 处的梯度")
    print(f"数值梯度: [sin(x) * cos(y), -cos(x) * sin(y)] = {numerical_gradient(complex_func, test_point)}")
    print("理论梯度:  [sin(x) * cos(y), -cos(x) * sin(y)] = [cos(π/4)*cos(π/3), -sin(π/4)*sin(π/3)]")

    # 测试6: 梯度下降示例
    print("\n6. 梯度下降法测试准备:")
    print("函数: f(x,y) = x^2 + y^2")
    print("初始点: (3.0, 4.0)")
    print("学习率: 0.1")
    print("迭代次数: 100")
    X, X_history = gradient_descent(func2d, np.array([3.0, 4.0]), lr=0.1, step_num=100)
    print(f"最终点: {X}")
    # print(f"过程历史: {X_history}")
    print("期望: 应收敛到最小值点 (0, 0) 附近")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 50)
    print("测试边界情况")
    print("=" * 50)

    print("1. 测试零向量:")
    zero_vector = np.array([0.0, 0.0, 0.0])

    print("\n2. 测试包含零的函数:")

    def func_with_zero(x):
        return 0.0  # 常数函数

    print("\n3. 测试非常大和非常小的值:")
    large_vector = np.array([1e10, 1e10])
    small_vector = np.array([1e-10, 1e-10])

    print("\n4. 测试不同h值的影响:")
    print("比较 h=1e-4, h=1e-6, h=1e-8 的精度")

    print("\n5. 测试函数不连续的情况:")

    def discontinuous_func(x):
        return 0 if x < 0 else 1

    print("\n注意: 数值微分对不连续函数效果不好")


def run_all_tests():
    """运行所有测试"""
    print("数值微分与梯度计算库测试")
    print("=" * 50)

    try:
        # test_numerical_diff()
        test_numerical_gradient()
        test_edge_cases()

        print("\n" + "=" * 50)
        print("测试用例定义完成")
        print("请先实现以下函数:")
        print("1. numerical_diff0(f, x)")
        print("2. numerical_diff(f, x)")
        print("3. _numerical_gradient(f, x)")
        print("4. numerical_gradient(f, X)")
        print("=" * 50)
        print("然后运行测试验证实现正确性")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        print("可能是函数未正确实现")


if __name__ == "__main__":
    run_all_tests()
