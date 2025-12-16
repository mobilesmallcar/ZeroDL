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


# ============================================================================
# 测试代码
# ============================================================================

def test_relu_layer():
    """测试ReLU层"""
    print("=" * 50)
    print("测试ReLU层")
    print("=" * 50)

    print("\n1. 测试前向传播:")
    relu = Relu()
    test_input = np.array([[-1.0, 0.0, 1.0],
                           [-2.0, 2.0, -3.0]])
    print(f"输入形状: {test_input.shape}")
    print(f"输入值:\n{test_input}")
    print(f"输出:\n{relu.forward(test_input)}")
    print("期望输出:\n[[0., 0., 1.],\n [0., 2., 0.]]")

    print("\n2. 测试反向传播:")
    upstream_grad = np.array([[1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]])
    print(f"上游梯度:\n{upstream_grad}")
    print(f"输出:\n{relu.backward(upstream_grad)}")
    print("期望输出梯度:\n[[0., 0., 1.],\n [0., 1., 0.]]")

    print("\n3. 测试mask属性:")
    print(f"mask: {relu.mask}")
    print("期望输出:\n [[True,True,False]\n[True,False,True]]")
    print("前向传播后，mask应记录哪些位置 <= 0")
    print("mask应与输入形状相同，值为布尔类型")


def test_sigmoid_layer():
    """测试Sigmoid层"""
    print("\n" + "=" * 50)
    print("测试Sigmoid层")
    print("=" * 50)

    print("\n1. 测试前向传播:")
    sigmoid = Sigmoid()
    test_input = np.array([-1.0, 0.0, 1.0])
    print(f"输入: {test_input}")
    print("期望输出: 所有值在(0, 1)范围内")
    print(f"输出:{sigmoid.forward(test_input)}")
    print("具体值应接近: [0.2689, 0.5, 0.7311]")

    print("\n2. 测试y属性保存:")
    print("前向传播后，self.y应保存输出值")
    print(f"输出:{sigmoid.y}")
    print("具体值应接近: [0.2689, 0.5, 0.7311]")
    print("用于反向传播计算")

    print("\n3. 测试反向传播:")
    upstream_grad = np.array([0.1, 0.2, 0.3])
    print(f"上游梯度: {upstream_grad}")
    print("公式: dx = dout * y * (1 - y)")
    print(f"输出:{sigmoid.backward(upstream_grad)}")
    print("具体值应接近: [0.01966119 0.05       0.05898358]")

    print("需要检查数值是否正确")


def test_affine_layer():
    """测试仿射层 - 使用更有代表性的值"""
    print("=" * 60)
    print("测试Affine层 - 详细计算验证")
    print("=" * 60)

    # ==================== 1. 定义测试数据 ====================
    W1 = np.array([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])
    b1 = np.array([0.1, 0.2, 0.3])
    affine1 = Affine(W1, b1)

    X1 = np.array([[1.0, 2.0],
                   [3.0, 4.0]])

    print("【测试数据】")
    print(f"权重W (2个输入特征×3个输出特征):\n{W1}")
    print(f"偏置b (3个输出特征): {b1}")
    print(f"输入X (2个样本×2个特征):\n{X1}")
    print()

    # ==================== 2. 前向传播验证 ====================
    print("【前向传播验证】")
    Y1 = affine1.forward(X1)
    print(f"计算得到的Y (2×3):\n{Y1}")

    expected_Y = np.array([[9.1, 12.2, 15.3],
                           [19.1, 26.2, 33.3]])
    print(f"期望的Y:\n{expected_Y}")

    if np.allclose(Y1, expected_Y, atol=1e-7):
        print("✅ 前向传播正确！")
    else:
        print("❌ 前向传播错误！")
    print()

    # ==================== 3. 反向传播验证 ====================
    print("【反向传播验证】")
    # 使用更有代表性的dout值
    dout1 = np.array([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6]])
    print(f"上游梯度dout (2×3):\n{dout1}")
    print()

    dX1 = affine1.backward(dout1)

    # 验证 dW
    print("1. 验证 dW = Xᵀ · dout:")
    print(f"Xᵀ (转置后2×2):\n{X1.T}")
    print(f"dout (2×3):\n{dout1}")
    print(f"计算得到的dW (2×3):\n{affine1.dW}")

    expected_dW = np.array([[1.3, 1.7, 2.1],
                            [1.8, 2.4, 3.0]])
    print(f"期望的dW:\n{expected_dW}")

    if np.allclose(affine1.dW, expected_dW, atol=1e-7):
        print("✅ dW计算正确！")
    else:
        print("❌ dW计算错误！")
    print()

    # 验证 db
    print("2. 验证 db = sum(dout, axis=0):")
    print(f"沿axis=0求和: {np.sum(dout1, axis=0)}")
    print(f"计算得到的db: {affine1.db}")

    expected_db = np.array([0.5, 0.7, 0.9])
    print(f"期望的db: {expected_db}")

    if np.allclose(affine1.db, expected_db, atol=1e-7):
        print("✅ db计算正确！")
    else:
        print("❌ db计算错误！")
    print()

    # 验证 dX
    print("3. 验证 dX = dout · Wᵀ:")
    print(f"Wᵀ (转置后3×2):\n{W1.T}")
    print(f"dout (2×3) @ Wᵀ (3×2) = dX (2×2)")
    print(f"计算得到的dX:\n{dX1}")

    expected_dX = np.array([[1.4, 3.2],
                            [3.2, 7.7]])
    print(f"期望的dX:\n{expected_dX}")

    if np.allclose(dX1, expected_dX, atol=1e-7):
        print("✅ dX计算正确！")
    else:
        print("❌ dX计算错误！")
    print()

    # ==================== 4. 直观理解 ====================
    print("【直观理解】")
    print("1. dW 的形状 (2,3):")
    print("   - 第1行: 输入特征1 对 3个输出特征的梯度")
    print("   - 第2行: 输入特征2 对 3个输出特征的梯度")
    print()

    print("2. db 的形状 (3,):")
    print("   - 每个输出特征对应的偏置梯度")
    print()

    print("3. dX 的形状 (2,2):")
    print("   - 第1行: 样本1的2个输入特征的梯度")
    print("   - 第2行: 样本2的2个输入特征的梯度")
    print()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


def test_softmax_with_loss():
    """测试SoftmaxWithLoss层"""
    print("\n" + "=" * 60)
    print("测试SoftmaxWithLoss层 - 详细计算验证")
    print("=" * 60)

    # ==================== 1. 测试基本功能 ====================
    print("\n【测试1: 基本功能验证】")
    layer = SoftmaxWithLoss()

    # 测试数据
    x = np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0]])

    # 测试one-hot编码
    print("1. one-hot编码标签测试:")
    t_onehot = np.array([[0, 0, 1],
                         [0, 1, 0]])
    print(f"输入x (2个样本, 3个类别):\n{x}")
    print(f"one-hot标签t:\n{t_onehot}")

    # 前向传播
    loss_onehot = layer.forward(x, t_onehot)
    print(f"计算得到的损失: {loss_onehot:.6f}")

    # 计算期望的softmax输出
    print("\n【手动计算验证】")
    print("1. 计算softmax:")
    print("样本1: [1.0, 2.0, 3.0]")
    x1_np = np.array([1.0, 2.0, 3.0])
    exp_x1 = np.exp(x1_np - 3.0)  # 减去最大值3.0
    softmax1 = exp_x1 / np.sum(exp_x1)
    print(f"   原始值: {x1_np}")
    print(f"   减去最大值3.0后: {x1_np - 3.0}")
    print(f"   exp(x1-max): {exp_x1}")
    print(f"   softmax(x1): {softmax1}")
    print(f"   概率和: {np.sum(softmax1):.6f} (应=1)")

    print("\n样本2: [2.0, 4.0, 6.0]")
    x2_np = np.array([2.0, 4.0, 6.0])
    exp_x2 = np.exp(x2_np - 6.0)  # 减去最大值6.0
    softmax2 = exp_x2 / np.sum(exp_x2)
    print(f"   原始值: {x2_np}")
    print(f"   减去最大值6.0后: {x2_np - 6.0}")
    print(f"   exp(x2-max): {exp_x2}")
    print(f"   softmax(x2): {softmax2}")
    print(f"   概率和: {np.sum(softmax2):.6f} (应=1)")

    # 计算期望损失
    print("\n2. 计算交叉熵损失:")
    print("  公式: L = -1/N Σ log(y_correct)")
    loss1 = -np.log(softmax1[2])  # 样本1的正确类别是索引2
    loss2 = -np.log(softmax2[1])  # 样本2的正确类别是索引1
    expected_loss = (loss1 + loss2) / 2
    print(f"   样本1正确类别概率 y[2] = {softmax1[2]:.6f}")
    print(f"   样本1损失: -log({softmax1[2]:.6f}) = {loss1:.6f}")
    print(f"   样本2正确类别概率 y[1] = {softmax2[1]:.6f}")
    print(f"   样本2损失: -log({softmax2[1]:.6f}) = {loss2:.6f}")
    print(f"   平均损失: ({loss1:.6f} + {loss2:.6f}) / 2 = {expected_loss:.6f}")
    print(f"   计算得到的损失: {loss_onehot:.6f}")

    if np.allclose(loss_onehot, expected_loss, atol=1e-7):
        print("✅ 前向传播正确！")
    else:
        print("❌ 前向传播错误！")
        print(f"差值: {loss_onehot - expected_loss}")

    # 验证layer中保存的y是否正确
    print(f"\n3. 验证layer保存的softmax输出:")
    print(f"   样本1 (期望: {softmax1}):")
    print(f"       实际: {layer.y[0]}")
    if np.allclose(layer.y[0], softmax1, atol=1e-7):
        print("       ✅ 正确")
    else:
        print("       ❌ 错误")

    print(f"   样本2 (期望: {softmax2}):")
    print(f"       实际: {layer.y[1]}")
    if np.allclose(layer.y[1], softmax2, atol=1e-7):
        print("       ✅ 正确")
    else:
        print("       ❌ 错误")

    # 反向传播
    print("\n4. one-hot编码反向传播:")
    print("  公式: dx = (y - t) / batch_size")
    print(f"  batch_size = 2")
    dx_onehot = layer.backward()
    print(f"计算得到的dx:\n{dx_onehot}")

    # 计算期望的dx
    expected_softmax = np.array([softmax1, softmax2])
    print(f"y (softmax输出):\n{expected_softmax}")
    print(f"t (one-hot标签):\n{t_onehot}")
    print(f"y - t:\n{expected_softmax - t_onehot}")
    expected_dx_onehot = (expected_softmax - t_onehot) / 2  # batch_size=2
    print(f"期望的dx (y - t)/2:\n{expected_dx_onehot}")

    if np.allclose(dx_onehot, expected_dx_onehot, atol=1e-7):
        print("✅ one-hot反向传播正确！")
    else:
        print("❌ one-hot反向传播错误！")
        print(f"差值:\n{dx_onehot - expected_dx_onehot}")

    # ==================== 2. 测试类别索引标签 ====================
    print("\n\n【测试2: 类别索引标签测试】")
    layer2 = SoftmaxWithLoss()

    t_indices = np.array([2, 1])  # 正确类别索引
    print(f"类别索引标签: {t_indices}")
    print(f"（对应one-hot: [[0,0,1], [0,1,0]]）")

    # 前向传播
    loss_indices = layer2.forward(x, t_indices)
    print(f"计算得到的损失: {loss_indices:.6f}")

    # 应该与one-hot编码得到相同损失
    if np.allclose(loss_indices, loss_onehot, atol=1e-7):
        print("✅ 类别索引与one-hot编码损失相同！")
    else:
        print("❌ 类别索引与one-hot编码损失不同！")
        print(f"差值: {loss_indices - loss_onehot}")

    # 反向传播
    print("\n类别索引反向传播:")
    print("  公式: dx = y.copy(); y[n,t] -= 1; dx /= batch_size")
    print(f"  batch_size = 2")
    dx_indices = layer2.backward()
    print(f"计算得到的dx:\n{dx_indices}")

    # 手动计算期望的dx
    expected_dx_indices = expected_softmax.copy()
    print(f"1. 复制y:\n{expected_dx_indices}")
    expected_dx_indices[0, 2] -= 1  # 样本1，正确类别2
    expected_dx_indices[1, 1] -= 1  # 样本2，正确类别1
    print(f"2. y[n,t] -= 1后:\n{expected_dx_indices}")
    expected_dx_indices /= 2  # 除以batch_size
    print(f"3. 除以batch_size=2后:\n{expected_dx_indices}")

    print(f"期望的dx:\n{expected_dx_indices}")

    if np.allclose(dx_indices, expected_dx_indices, atol=1e-7):
        print("✅ 类别索引反向传播正确！")
    else:
        print("❌ 类别索引反向传播错误！")
        print(f"差值:\n{dx_indices - expected_dx_indices}")

    # ==================== 3. 测试数值稳定性 ====================
    # print("\n\n【测试3: 数值稳定性测试】")
    # layer3 = SoftmaxWithLoss()
    #
    # # 大数值输入测试
    # x_large = np.array([[1000.0, 1001.0, 1002.0]])
    # t_large = np.array([2])  # 正确类别是索引2
    #
    # print(f"大数值输入 (可能引起exp溢出): {x_large}")
    # print(f"标签: {t_large}")
    # print("注意: 直接计算exp(1000)会溢出，但减去最大值后exp(0)=1")
    #
    # try:
    #     loss_large = layer3.forward(x_large, t_large)
    #     dx_large = layer3.backward()
    #
    #     print(f"计算得到的损失: {loss_large:.6f}")
    #     print(f"反向传播dx: {dx_large}")
    #
    #     # 手动验证
    #     print("\n手动验证:")
    #     x_large_np = np.array([1000.0, 1001.0, 1002.0])
    #     exp_large = np.exp(x_large_np - 1002.0)  # 减去最大值1002.0
    #     softmax_large = exp_large / np.sum(exp_large)
    #     print(f"减去最大值1002.0后: {x_large_np - 1002.0}")
    #     print(f"exp值: {exp_large}")
    #     print(f"softmax: {softmax_large}")
    #     print(f"概率和: {np.sum(softmax_large):.6f}")
    #
    #     expected_loss_large = -np.log(softmax_large[2])
    #     print(f"期望损失: {expected_loss_large:.6f}")
    #
    #     # 检查是否出现数值异常
    #     if np.any(np.isnan(loss_large)) or np.any(np.isinf(loss_large)):
    #         print("❌ 数值溢出！损失包含NaN或Inf")
    #     elif np.any(np.isnan(dx_large)) or np.any(np.isinf(dx_large)):
    #         print("❌ 数值溢出！梯度包含NaN或Inf")
    #     elif np.allclose(loss_large, expected_loss_large, atol=1e-7):
    #         print("✅ 数值稳定性良好！计算结果正确")
    #     else:
    #         print("⚠️  数值稳定但计算结果有差异")
    #         print(f"差值: {loss_large - expected_loss_large}")
    #
    # except Exception as e:
    #     print(f"❌ 数值稳定性测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()

    # ==================== 4. 测试边界情况 ====================
    print("\n\n【测试4: 边界情况测试】")

    # 4.1 完美预测情况
    print("4.1 完美预测测试:")
    layer4 = SoftmaxWithLoss()
    x_perfect = np.array([[0.0, 0.0, 10.0],  # 第三个类别概率接近1
                          [0.0, 10.0, 0.0]])  # 第二个类别概率接近1
    t_perfect = np.array([2, 1])

    loss_perfect = layer4.forward(x_perfect, t_perfect)
    print(f"输入（完美预测）:\n{x_perfect}")
    print(f"标签: {t_perfect}")
    print(f"损失: {loss_perfect:.6f}")

    # 计算softmax
    exp_perfect1 = np.exp(np.array([0.0, 0.0, 10.0]) - 10.0)
    softmax_perfect1 = exp_perfect1 / np.sum(exp_perfect1)
    print(f"样本1 softmax: {softmax_perfect1}")
    print(f"正确类别概率: {softmax_perfect1[2]:.6f}")

    if loss_perfect < 0.001:
        print("✅ 完美预测损失接近0！")
    else:
        print(f"⚠️  完美预测损失不接近0: {loss_perfect}")

    # 4.2 单样本测试
    print("\n4.2 单样本测试:")
    layer6 = SoftmaxWithLoss()
    x_single = np.array([[1.0, 2.0, 3.0]])
    t_single = np.array([2])

    loss_single = layer6.forward(x_single, t_single)
    dx_single = layer6.backward()

    print(f"单样本输入: {x_single}")
    print(f"单样本标签: {t_single}")
    print(f"损失: {loss_single:.6f}")
    print(f"梯度dx: {dx_single}")

    # 手动计算期望
    exp_single = np.exp(np.array([1.0, 2.0, 3.0]) - 3.0)
    softmax_single = exp_single / np.sum(exp_single)
    expected_dx_single = softmax_single.copy()
    expected_dx_single[2] -= 1  # batch_size=1，所以直接y-t
    print(f"期望dx: {expected_dx_single}")

    if np.allclose(dx_single[0], expected_dx_single, atol=1e-7):
        print("✅ 单样本测试正确！")
    else:
        print("❌ 单样本测试错误！")

    # ==================== 5. 属性保存验证 ====================
    print("\n\n【测试5: 属性保存验证】")
    layer_test = SoftmaxWithLoss()
    x_test = np.array([[1.0, 2.0, 3.0]])
    t_test = np.array([2])

    print("调用forward前:")
    print(f"  self.y: {layer_test.y}")
    print(f"  self.t: {layer_test.t}")
    print(f"  self.loss: {layer_test.loss}")

    loss_test = layer_test.forward(x_test, t_test)

    print("\n调用forward后:")
    print("1. self.y (softmax输出):")
    print(f"   值: {layer_test.y}")
    print(f"   形状: {layer_test.y.shape}")
    print(f"   概率和: {np.sum(layer_test.y):.6f} (应=1)")

    print(f"\n2. self.t (目标标签): {layer_test.t}")
    print(f"   类型: {type(layer_test.t)}")
    print(f"   值: {layer_test.t}")

    print(f"\n3. self.loss (损失值): {layer_test.loss:.6f}")

    # 验证属性是否正确保存
    properties_ok = True
    if layer_test.y is None:
        print("❌ self.y 未保存！")
        properties_ok = False
    if layer_test.t is None:
        print("❌ self.t 未保存！")
        properties_ok = False
    if layer_test.loss is None:
        print("❌ self.loss 未保存！")
        properties_ok = False

    if properties_ok:
        print("✅ 所有属性正确保存！")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


def run_all_tests():
    """运行所有测试"""
    print("神经网络层测试")
    print("=" * 50)

    # try:
    test_relu_layer()
    test_sigmoid_layer()
    test_affine_layer()
    test_softmax_with_loss()

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

    # except Exception as e:
    #     print(f"\n测试过程中出现错误: {e}")
    #     print("可能是函数未正确实现或导入错误")


if __name__ == "__main__":
    run_all_tests()
