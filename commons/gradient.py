# 数值微分与梯度计算库
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
        x_history.append(x.copy())  # 保存当前点到列表
        grad = numerical_gradient(f, x)  # 计算梯度
        x = x - lr * grad  # 更新点
    return x, np.array(x_history)
