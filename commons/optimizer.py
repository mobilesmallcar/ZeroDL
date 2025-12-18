# 优化算法库
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

作者: xsy
版本: 1.0
"""

import numpy as np


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
    SGD (Stochastic Gradient Descent) 随机梯度下降
    更新公式：
        1. W ← W - α * ∇          # 参数沿梯度负方向更新

        - W: 模型参数
        - ∇: 当前梯度
        - α: 学习率（控制步长）

    特性:
        - 最基础优化算法：直接利用梯度最小化损失
        - 随机性：mini-batch 引入噪声，有助于逃离局部最优
        - 计算高效：适合大规模数据
        - 震荡收敛：路径锯齿状，但整体下降
        - 学习率敏感：α 过大易发散，过小收敛慢
        - 无自适应：所有参数统一学习率
        - 基础性强：现代优化器（如Momentum、Adam）均以此为基础
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
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    Momentum 动量法
    更新公式：
        减法版本
        1. v ← a * v + η * ∇          # 速度累积：当前梯度 + 历史速度
        2. W ← W - v                  # 参数更新：沿累积速度方向移动
        加法版本
        1. v ← a × v  +  (1 - a) × ∇      # 速度 = 保留旧速度 + 一小部分当前梯度
        2. W ← W  -  η × v               # 参数向（负）速度方向移动（因为要下山）
        - v: 速度（动量项）
        - ∇: 当前梯度即 ∂L/∂W
        - a: 动量系数（通常 0.9）
        - η: 学习率

    特性:
        - 加速收敛：累积历史梯度，平缓方向加速、陡峭方向减速
        - 抑制震荡：穿越鞍部与局部谷地更平稳
        - 路径平滑：更新轨迹更直接接近最优
        - 简单高效：仅需存储速度v
        - 适合复杂损失景观：比纯SGD更快收敛
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
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    AdaGrad (Adaptive Gradient) 自适应梯度算法
    更新公式：
        1. h ← h + ∇²                 # 累积梯度平方
        2. W ← W - α * ∇ / (√h + ε)   # 自适应更新：历史梯度越大步长越小

        - h: 梯度平方的累积和
        - ∇: 当前梯度
        - α: 初始学习率
        - ε: 防除零小常数（1e-7）

    特性:
        - 自适应学习率：频繁参数学习率快速衰减，稀疏参数保持较大步长
        - 适合稀疏数据：特征出现频率差异大时效果好
        - 无需手动调学习率：自动调整每个参数步长
        - 学习率单调衰减：后期可能过小导致收敛停滞
        - 内存高效：仅需存储h
        - 基础性强：RMSProp、Adam等后续优化器的灵感来源
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
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] ** 2
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSProp:
    """
    RMSProp (Root Mean Square Propagation) 均方根传播
    更新公式：
        1. h ← α * h + (1 - α) * ∇²         # 梯度平方的指数移动平均
        2. W ← W - η * ∇ / √(h + ε)         # 自适应更新

        - h: 梯度平方的移动平均
        - ∇: 当前梯度
        - α: 衰减率（通常 0.99）
        - η: 学习率
        - ε: 防除零小常数（1e-7）

    特性:
        - 自适应学习率：梯度大时步长小，梯度小时步长大
        - 缓解AdaGrad学习率衰减过快问题
        - 抑制震荡、加速收敛
        - 适合非平稳与稀疏梯度场景
        - 内存高效：仅需存储h
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
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] *= self.alpha
            self.h[key] += (1 - self.alpha) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:
    """
    Adam (Adaptive Moment Estimation) 自适应矩估计
    更新公式：
        1. v ← α1 * v + (1 - α1) * ∇          # 一阶矩（动量）
        2. h ← α2 * h + (1 - α2) * ∇²         # 二阶矩（自适应）
        3. \\hat{v} ← v / (1 - α1^t)           # 一阶偏差修正
        4. \\hat{h} ← h / (1 - α2^t)           # 二阶偏差修正
        5. W ← W - η * \\hat{v} / √(\\hat{h} + ε)   # 参数更新

        - v: 一阶矩（梯度均值）
        - h: 二阶矩（梯度平方均值）
        - α1: 一阶衰减率（通常 0.9）
        - α2: 二阶衰减率（通常 0.999）
        - η: 学习率
        - ε: 防除零小常数（1e-8）
        - t: 更新步数

    特性:
        - 融合动量与RMSProp：方向稳定 + 自适应步长
        - 偏差修正：早期训练更稳定
        - 收敛快、鲁棒性强：适合噪声/稀疏梯度
        - 超参数友好：默认值表现优秀
        - 内存需求：每个参数存v和h
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
        if self.v is None:
            self.v, self.h = {}, {}
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])
                self.h[key] = np.zeros_like(params[key])
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.alpha2**self.t) / (1.0 - self.alpha1**self.t)
        # 更新参数
        for key in params.keys():
            # self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            # self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * (grads[key]**2)
            self.v[key] += (1 - self.alpha1) * (grads[key] - self.v[key])
            self.h[key] += (1 - self.alpha2) * (grads[key]**2 - self.h[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-7)