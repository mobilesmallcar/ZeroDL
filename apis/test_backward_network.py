# 两层神经网络用反向传播实现--调用函数
import math
import numpy as np
from _03_two_layerNet_backward import TwoLayerNet

from commons.load_data import get_data

# 1.加载数据
x_train, x_test, y_train, y_test = get_data()
print('x_train shape:', x_train.shape)
print('t_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('t_test shape:', y_test.shape)
# 2.创建神经网络
net = TwoLayerNet(input_size=784, hidden_size=20, output_size=10, weight_init_std=0.01)

# 3.设置超参数
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = math.ceil(train_size / batch_size)

iters_num = 10000
learning_rate = 0.1

train_acc_list = []
test_acc_list = []
train_loss_list = []

# 4.随机梯度下降法，迭代训练模型，计算参数
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = y_train[batch_mask]

    # 计算梯度
    grads = net.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grads[key]

    # 计算损失
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # print(f"iter {i},loss {loss}")

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, y_train)
        test_acc = net.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"rain loss: {loss}, train_acc {train_acc},test_acc {test_acc}")
