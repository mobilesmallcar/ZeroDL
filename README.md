# 手搓神经网络：从零开始的深度学习实践

## 📖 项目概述

本项目是一个从零实现神经网络底层组件的机器学习学习仓库，参考《深度学习入门：基于Python的理论与实现》（斋藤康毅）等经典教材。

逐步实现了激活函数、损失函数、层结构、数值微分/梯度检查、优化器等核心组件，最终构建并训练了多层神经网络，能够在 MNIST 手写数字数据集上达到 **97%+** 的识别准确率。

项目重点在于理解神经网络的数学原理与实现细节，所有代码均使用纯 NumPy 实现（不依赖 PyTorch/TensorFlow 等高级框架），便于初学者逐行阅读和调试。

## 🌿 分支说明

项目采用多分支开发策略，主要分支如下：

- **master**  
  主分支，包含稳定、可运行的完整代码。  
  所有经过验证的实现都会最终合并到这里。  
  **推荐直接使用 master 或 dev 分支，二者任选其一即可**。

- **dev**  
  实验性分支，用于开发过程中的快速验证与调试。  
  特点：**测试代码直接写在每个类或函数下方**（未集中到独立的测试文件或测试框架），便于边实现边测试、即时查看效果。  
  该分支提交频繁，历史记录多为小步尝试与临时调试，不追求代码整洁度。  
  **仅适合开发阶段参考，不建议作为最终学习或运行代码的来源**。
- 
## 🎯 核心目标

1. **实现意义**：理解神经网络每个组件的数学原理和实现细节
2. **实践价值**：掌握如何从零构建一个可用的深度学习框架
3. **代码质量**：编写模块化、可测试、文档完善的代码

## 📁 项目结构

```
neural_network_from_scratch/
│   ├── common/
│   │   ├── functions.py          # 激活函数和损失函数
│   │   ├── layers.py             # 神经网络层实现
│   │   ├── optimizers.py         # 优化算法
│   │   └── test_all.py           # 统一测试框架
│   ├── api/  # 调用层
└── README.md             # 项目文档

```

## 🔧 核心组件

### 1. **激活函数** (`functions.py`)
   - `step_function` - 阶跃函数
   - `sigmoid` - Sigmoid激活函数
   - `relu` - ReLU激活函数
   - `softmax` - Softmax函数（用于多分类）
   - `identity` - 恒等函数（用于回归）

### 2. **损失函数** (`functions.py`)
   - `mean_squared_error` - 均方误差（回归问题）
   - `cross_entropy_error` - 交叉熵误差（分类问题）

### 3. **数值计算** (`numerical_gradient.py`)
   - `numerical_diff` - 数值微分（中心差分法）
   - `numerical_gradient` - 数值梯度计算
   - 用于验证反向传播的正确性

### 4. **神经网络层** (`layers.py`)
   - `Relu` - ReLU激活层（前向传播 + 反向传播）
   - `Sigmoid` - Sigmoid激活层
   - `Affine` - 全连接层（仿射变换）
   - `SoftmaxWithLoss` - Softmax + 交叉熵损失层

### 5. **优化算法** (`optimizers.py`)
   - `SGD` - 随机梯度下降
   - `Momentum` - 动量法
   - `AdaGrad` - 自适应梯度算法
   - `RMSProp` - 均方根传播
   - `Adam` - 自适应矩估计

## 🧪 测试框架 (`test_all.py`)

提供了一个完整的测试套件，可以：
- 单独测试每个组件
- 验证前向/反向传播的正确性
- 检查梯度计算的准确性
- 模拟完整的神经网络训练流程

## 📚 数学原理

### 前向传播
```
输入X → 仿射变换 → 激活函数 → ... → 输出层 → 损失计算
```

### 反向传播（链式法则）
```
损失L → ∂L/∂输出 → ... → ∂L/∂权重 → 优化器更新
```

### 关键公式
1. **仿射变换**：Y = XW + b
2. **Sigmoid导数**：σ'(x) = σ(x)(1-σ(x))
3. **ReLU导数**：f'(x) = 1 if x>0 else 0
4. **交叉熵梯度**：∂L/∂y = (y - t)/batch_size

## 🚀 如何使用

### 1. 实现所有组件
```python
# 先实现所有TODO标记的函数
# 按照以下顺序：
# 1. functions.py 中的激活函数和损失函数
# 2. numerical_gradient.py 中的数值微分
# 3. layers.py 中的神经网络层
# 4. optimizers.py 中的优化算法
```

### 2. 运行测试
```bash
# 到项目根目录
cd */ml-learninng
# 测试所有组件
python -m commons.test_all_common

# 只测试神经网络层
python -m apis.test_backward_network
python -m apis.test_simple_numerical_network
```

### 3. 构建神经网络示例
```python
# 创建一个两层神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 训练循环
for epoch in range(max_epochs):
    # 前向传播
    loss = network.loss(x_batch, t_batch)
    
    # 反向传播
    grads = network.gradient(x_batch, t_batch)
    
    # 优化器更新
    optimizer.update(network.params, grads)
```


## 🛠️ 技术栈

- **Python 3.12+**：主要编程语言
- **NumPy**：核心数值计算库
- **纯数学实现**：不依赖任何深度学习框架

这个项目是理解深度学习底层原理的绝佳实践，通过"手搓"每个组件，你将获得对神经网络工作原理的深刻理解，这是使用高级框架无法替代的学习体验！

