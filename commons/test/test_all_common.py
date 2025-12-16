# 神经网络组件统一测试框架-自定义实现
"""
神经网络组件统一测试框架
================================

本模块提供统一的测试框架，用于测试所有神经网络组件：
1. 激活函数和损失函数
2. 数值微分和梯度计算
3. 神经网络层（前向/反向传播）
4. 优化算法

使用说明:
    python test_all.py    # 运行所有测试
    python test_all.py --test activation  # 只测试激活函数
    python test_all.py --test layers      # 只测试神经网络层

作者: 测试框架开发者
版本: 1.0
"""

import numpy as np
import argparse
import sys
import time

# ============================================================================
# 导入待测试的模块（根据实际文件名调整）
# ============================================================================

from commons.functions import ActivationFunctions, LossFunctions
from commons.layer import Relu, Sigmoid, Affine, SoftmaxWithLoss
from commons.optimizer import SGD, Momentum, AdaGrad, RMSProp, Adam
from commons.gradient import numerical_diff, numerical_gradient


class NeuralNetworkTestSuite:
    """神经网络组件统一测试套件"""

    def __init__(self, verbose=True):
        """
        初始化测试套件

        参数:
            verbose: 是否打印详细测试信息
        """
        self.verbose = verbose
        self.passed_tests = 0
        self.failed_tests = 0
        self.start_time = None

    def _print_test_header(self, title):
        """打印测试标题"""
        print(f"\n{'=' * 60}")
        print(f"测试: {title}")
        print(f"{'=' * 60}")

    def _assert(self, condition, message):
        """自定义断言，记录测试结果"""
        if condition:
            if self.verbose:
                print(f"  ✅ {message}")
            self.passed_tests += 1
        else:
            print(f"  ❌ {message}")
            self.failed_tests += 1
        return condition

    def _test_value_close(self, actual, expected, tolerance=1e-7, message=""):
        """测试数值是否接近"""
        if np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
            if self.verbose:
                print(f"  ✅ {message} (实际: {actual}, 期望: {expected})")
            self.passed_tests += 1
            return True
        else:
            print(f"  ❌ {message} (实际: {actual}, 期望: {expected})")
            self.failed_tests += 1
            return False

    def _test_shape_match(self, actual, expected_shape, message=""):
        """测试形状是否匹配"""
        if actual.shape == expected_shape:
            if self.verbose:
                print(f"  ✅ {message} (形状: {actual.shape})")
            self.passed_tests += 1
            return True
        else:
            print(f"  ❌ {message} (实际形状: {actual.shape}, 期望: {expected_shape})")
            self.failed_tests += 1
            return False

    def start(self):
        """开始测试"""
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("神经网络组件测试套件")
        print("=" * 60)
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def finish(self):
        """结束测试，打印统计信息"""
        elapsed_time = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"总测试数: {self.passed_tests + self.failed_tests}")
        print(f"通过: {self.passed_tests}")
        print(f"失败: {self.failed_tests}")
        print(f"成功率: {self.passed_tests / (self.passed_tests + self.failed_tests) * 100:.1f}%")
        print(f"用时: {elapsed_time:.2f}秒")
        print("=" * 60)

        if self.failed_tests == 0:
            print("🎉 所有测试通过！")
            return True
        else:
            print("⚠️  有测试失败，请检查实现")
            return False

    # ============================================================================
    # 激活函数和损失函数测试
    # ============================================================================

    def test_activation_functions(self):
        """测试激活函数"""
        self._print_test_header("激活函数")

        # 测试阶跃函数
        x_step = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected_step = np.array([0, 0, 1, 1, 1])

        try:
            result = ActivationFunctions.step_function(x_step)
            self._test_value_close(result, expected_step, 1e-7, "阶跃函数")
        except Exception as e:
            self._assert(False, f"阶跃函数异常: {e}")

        # 测试Sigmoid函数
        x_sigmoid = np.array([-1.0, 0.0, 1.0])

        try:
            result = ActivationFunctions.sigmoid(x_sigmoid)
            # 检查值范围
            self._assert(np.all(result > 0) and np.all(result < 1), "Sigmoid输出范围(0,1)")
            # 检查对称性
            result_neg = ActivationFunctions.sigmoid(-x_sigmoid)
            self._test_value_close(result + result_neg, np.ones_like(result), 1e-7, "Sigmoid对称性")
        except Exception as e:
            self._assert(False, f"Sigmoid函数异常: {e}")

        # 测试ReLU函数
        x_relu = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        expected_relu = np.array([0, 0, 0, 1, 3])

        try:
            result = ActivationFunctions.relu(x_relu)
            self._test_value_close(result, expected_relu, 1e-7, "ReLU函数")
        except Exception as e:
            self._assert(False, f"ReLU函数异常: {e}")

        # 测试Softmax函数
        x_softmax = np.array([1.0, 2.0, 3.0])

        try:
            result = ActivationFunctions.softmax(x_softmax)
            # 检查概率和为1
            self._test_value_close(np.sum(result), 1.0, 1e-7, "Softmax概率和")
            # 检查最大概率对应最大输入
            self._assert(np.argmax(result) == np.argmax(x_softmax), "Softmax最大值对应")
        except Exception as e:
            self._assert(False, f"Softmax函数异常: {e}")

        # 测试恒等函数
        x_identity = np.array([1.0, 2.0, 3.0])

        try:
            result = ActivationFunctions.identity(x_identity)
            self._test_value_close(result, x_identity, 1e-7, "恒等函数")
        except Exception as e:
            self._assert(False, f"恒等函数异常: {e}")

    def test_loss_functions(self):
        """测试损失函数"""
        self._print_test_header("损失函数")

        # 测试均方误差
        y_pred = np.array([0.1, 0.2, 0.3])
        y_true = np.array([0.1, 0.2, 0.3])

        try:
            result = LossFunctions.mean_squared_error(y_pred, y_true)
            self._test_value_close(result, 0.0, 1e-7, "MSE完美预测")
        except Exception as e:
            self._assert(False, f"MSE异常: {e}")

        # 测试交叉熵误差（one-hot编码）
        y_pred_ce = np.array([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
        y_true_onehot = np.array([[0, 1, 0], [0, 0, 1]])

        try:
            result = LossFunctions.cross_entropy_error(y_pred_ce, y_true_onehot)
            self._assert(result > 0, "交叉熵损失为正")
            # 测试数值稳定性
            y_pred_edge = np.array([[0.0, 0.0, 1.0]])
            y_true_edge = np.array([[0, 0, 1]])
            result_edge = LossFunctions.cross_entropy_error(y_pred_edge, y_true_edge)
            self._assert(not np.isnan(result_edge), "交叉熵数值稳定性")
        except Exception as e:
            self._assert(False, f"交叉熵异常: {e}")

    # ============================================================================
    # 数值微分和梯度测试
    # ============================================================================

    def test_numerical_differentiation(self):
        """测试数值微分"""
        self._print_test_header("数值微分")

        # 测试函数 f(x) = x^2
        def f_square(x):
            return x ** 2

        try:
            # 测试点 x=2, f'(x)=4

            result = numerical_diff(f_square, 2.0)
            self._test_value_close(result, 4.0, 1e-5, "数值微分x²")
        except Exception as e:
            self._assert(False, f"数值微分异常: {e}")

        # 测试函数 f(x) = sin(x)
        def f_sin(x):
            return np.sin(x)

        try:
            # 测试点 x=0, f'(x)=cos(0)=1
            result = numerical_diff(f_sin, 0.0)
            self._test_value_close(result, 1.0, 1e-5, "数值微分sin(x)")
        except Exception as e:
            self._assert(False, f"数值微分sin异常: {e}")

    def test_numerical_gradient(self):
        """测试数值梯度"""
        self._print_test_header("数值梯度")

        # 测试函数 f(x,y) = x^2 + y^2
        def f_sphere(x):
            return np.sum(x ** 2)

        try:

            test_point = np.array([3.0, 4.0])
            result = numerical_gradient(f_sphere, test_point)
            expected = np.array([6.0, 8.0])
            self._test_value_close(result, expected, 1e-5, "数值梯度x²+y²")
        except Exception as e:
            self._assert(False, f"数值梯度异常: {e}")

    # ============================================================================
    # 神经网络层测试
    # ============================================================================

    def test_relu_layer(self):
        """测试ReLU层"""
        self._print_test_header("ReLU层")

        try:
            relu = Relu()
            x = np.array([[-1.0, 0.0, 1.0], [-2.0, 2.0, -3.0]])

            # 前向传播
            y = relu.forward(x)
            expected_y = np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
            self._test_value_close(y, expected_y, 1e-7, "ReLU前向传播")

            # 反向传播
            dout = np.ones_like(x)
            dx = relu.backward(dout)
            expected_dx = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            self._test_value_close(dx, expected_dx, 1e-7, "ReLU反向传播")
        except Exception as e:
            self._assert(False, f"ReLU层异常: {e}")

    def test_sigmoid_layer(self):
        """测试Sigmoid层"""
        self._print_test_header("Sigmoid层")

        try:
            sigmoid = Sigmoid()
            x = np.array([-1.0, 0.0, 1.0])

            # 前向传播
            y = sigmoid.forward(x)
            self._assert(np.all(y > 0) and np.all(y < 1), "Sigmoid输出范围")

            # 反向传播
            dout = np.array([0.1, 0.2, 0.3])
            dx = sigmoid.backward(dout)
            self._test_shape_match(dx, x.shape, "Sigmoid梯度形状")
        except Exception as e:
            self._assert(False, f"Sigmoid层异常: {e}")

    def test_affine_layer(self):
        """测试仿射层"""
        self._print_test_header("仿射层")

        try:
            W = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            b = np.array([0.1, 0.2])
            affine = Affine(W, b)

            X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            # 前向传播
            Y = affine.forward(X)
            expected_shape = (2, 2)
            self._test_shape_match(Y, expected_shape, "Affine输出形状")

            # 反向传播
            dout = np.ones_like(Y)
            dX = affine.backward(dout)
            self._test_shape_match(dX, X.shape, "Affine梯度形状")
            self._assert(affine.dW is not None, "Affine权重梯度计算")
            self._assert(affine.db is not None, "Affine偏置梯度计算")
        except Exception as e:
            self._assert(False, f"仿射层异常: {e}")

    def test_softmax_with_loss(self):
        """测试SoftmaxWithLoss层"""
        self._print_test_header("SoftmaxWithLoss层")

        try:
            layer = SoftmaxWithLoss()

            # 测试one-hot编码
            x = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
            t_onehot = np.array([[0, 0, 1], [0, 1, 0]])

            loss = layer.forward(x, t_onehot)
            self._assert(loss > 0, "交叉熵损失为正")
            dx = layer.backward()
            self._test_shape_match(dx, x.shape, "SoftmaxWithLoss梯度形状")
        except Exception as e:
            self._assert(False, f"SoftmaxWithLoss层异常: {e}")

    # ============================================================================
    # 优化算法测试
    # ============================================================================

    def test_sgd_optimizer(self):
        """测试SGD优化器"""
        self._print_test_header("SGD优化器")

        try:
            sgd = SGD(lr=0.01)
            params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
            grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]])}
            initial_W = params['W'].copy()

            sgd.update(params, grads)
            expected_W = initial_W - 0.01 * grads['W']
            self._test_value_close(params['W'], expected_W, 1e-7, "SGD参数更新")
        except Exception as e:
            self._assert(False, f"SGD优化器异常: {e}")

    def test_momentum_optimizer(self):
        """测试Momentum优化器"""
        self._print_test_header("Momentum优化器")

        try:
            momentum = Momentum(lr=0.01, momentum=0.9)
            params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
            grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]])}

            # 第一次更新
            initial_W = params['W'].copy()
            momentum.update(params, grads)
            self._assert(momentum.v is not None, "Momentum初始化速度")
        except Exception as e:
            self._assert(False, f"Momentum优化器异常: {e}")

    def test_adagrad_optimizer(self):
        """测试AdaGrad优化器"""
        self._print_test_header("AdaGrad优化器")

        try:
            adagrad = AdaGrad(lr=0.01)
            params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
            grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]])}

            adagrad.update(params, grads)
            self._assert(adagrad.h is not None, "AdaGrad初始化h")
        except Exception as e:
            self._assert(False, f"AdaGrad优化器异常: {e}")

    def test_rmsprop_optimizer(self):
        """测试RMSProp优化器"""
        self._print_test_header("RMSProp优化器")

        try:
            rmsprop = RMSProp(lr=0.01, alpha=0.9)
            params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
            grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]])}

            rmsprop.update(params, grads)
            self._assert(rmsprop.h is not None, "RMSProp初始化h")
        except Exception as e:
            self._assert(False, f"RMSProp优化器异常: {e}")

    def test_adam_optimizer(self):
        """测试Adam优化器"""
        self._print_test_header("Adam优化器")

        try:
            adam = Adam(lr=0.01, alpha1=0.9, alpha2=0.999)
            params = {'W': np.array([[1.0, 2.0], [3.0, 4.0]])}
            grads = {'W': np.array([[0.1, 0.2], [0.3, 0.4]])}

            adam.update(params, grads)
            self._assert(adam.v is not None, "Adam初始化v")
            self._assert(adam.h is not None, "Adam初始化h")
            self._assert(adam.t == 1, "Adam时间步更新")
        except Exception as e:
            self._assert(False, f"Adam优化器异常: {e}")

    # ============================================================================
    # 集成测试
    # ============================================================================

    def test_neural_network_workflow(self):
        """测试完整神经网络工作流程"""
        self._print_test_header("神经网络工作流程")

        try:
            # 构建一个简单的两层网络
            np.random.seed(42)  # 固定随机种子

            # 参数初始化
            W1 = np.random.randn(3, 4)
            b1 = np.random.randn(4)
            W2 = np.random.randn(4, 2)
            b2 = np.random.randn(2)

            # 创建网络层
            affine1 = Affine(W1, b1)
            relu1 = Relu()
            affine2 = Affine(W2, b2)
            loss_layer = SoftmaxWithLoss()

            # 创建优化器
            optimizer = SGD(lr=0.01)

            # 模拟数据
            X = np.random.randn(5, 3)  # 5个样本，3个特征
            T = np.random.randint(0, 2, size=(5,))  # 二分类标签

            # 前向传播
            y1 = affine1.forward(X)
            a1 = relu1.forward(y1)
            y2 = affine2.forward(a1)
            loss = loss_layer.forward(y2, T)

            self._assert(loss > 0, "前向传播计算损失")

            # 反向传播
            dy2 = loss_layer.backward()
            da1 = affine2.backward(dy2)
            dy1 = relu1.backward(da1)
            _ = affine1.backward(dy1)

            # 准备梯度字典
            grads = {
                'W1': affine1.dW,
                'b1': affine1.db,
                'W2': affine2.dW,
                'b2': affine2.db
            }

            # 准备参数字典
            params = {
                'W1': W1,
                'b1': b1,
                'W2': W2,
                'b2': b2
            }

            # 优化器更新参数
            optimizer.update(params, grads)

            self._assert(True, "完整工作流程执行成功")

        except Exception as e:
            self._assert(False, f"工作流程异常: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================================
    # 运行测试组
    # ============================================================================

    def run_all_tests(self):
        """运行所有测试"""
        self.start()

        # 激活函数和损失函数
        self.test_activation_functions()
        self.test_loss_functions()

        # 数值微分和梯度
        self.test_numerical_differentiation()
        self.test_numerical_gradient()

        # 神经网络层
        self.test_relu_layer()
        self.test_sigmoid_layer()
        self.test_affine_layer()
        self.test_softmax_with_loss()

        # 优化算法
        self.test_sgd_optimizer()
        self.test_momentum_optimizer()
        self.test_adagrad_optimizer()
        self.test_rmsprop_optimizer()
        self.test_adam_optimizer()

        # 集成测试
        self.test_neural_network_workflow()

        return self.finish()

    def run_test_group(self, group_name):
        """运行指定测试组"""
        self.start()

        test_groups = {
            'activation': ['test_activation_functions', 'test_loss_functions'],
            'numerical': ['test_numerical_differentiation', 'test_numerical_gradient'],
            'layers': ['test_relu_layer', 'test_sigmoid_layer',
                       'test_affine_layer', 'test_softmax_with_loss'],
            'optimizers': ['test_sgd_optimizer', 'test_momentum_optimizer',
                           'test_adagrad_optimizer', 'test_rmsprop_optimizer',
                           'test_adam_optimizer'],
            'workflow': ['test_neural_network_workflow']
        }

        if group_name in test_groups:
            print(f"\n运行测试组: {group_name}")
            for test_method in test_groups[group_name]:
                getattr(self, test_method)()
        else:
            print(f"\n未知测试组: {group_name}")
            print("可用测试组:", list(test_groups.keys()))
            return False

        return self.finish()


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='神经网络组件测试框架')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'activation', 'numerical',
                                 'layers', 'optimizers', 'workflow'],
                        help='选择要运行的测试组')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细测试信息')
    parser.add_argument('--quiet', action='store_true',
                        help='只显示失败测试')

    args = parser.parse_args()

    # 设置详细程度
    if args.quiet:
        verbose = False
    else:
        verbose = args.verbose or True  # 默认显示详细信息

    # 创建测试套件
    test_suite = NeuralNetworkTestSuite(verbose=verbose)

    # 运行测试
    if args.test == 'all':
        success = test_suite.run_all_tests()
    else:
        success = test_suite.run_test_group(args.test)

    # 返回退出码
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
