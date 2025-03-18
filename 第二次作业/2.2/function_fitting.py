"""
2251545 邢致远
使用NumPy实现的两层ReLU神经网络
"""

import numpy as np
import matplotlib.pyplot as plt


class ReluNetwork:
    # 两层ReLU神经网络模型
    # mu表示动量系数，防止震荡  噪声标准差负责正则化
    def __init__(self, hidden_units=128, learning_rate=0.01, mu=0.9, noise_std=0.01, max_grad_norm=1.0):
        # 网络结构参数
        self.hidden_units = hidden_units

        # 网络权重参数
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # 优化参数
        self.learning_rate = learning_rate
        self.mu = mu
        self.noise_std = noise_std
        self.max_grad_norm = max_grad_norm

        # 数据标准化参数
        self.x_mean = None
        self.x_std = None

        # 动量累积变量
        self.vW1, self.vb1 = 0, 0
        self.vW2, self.vb2 = 0, 0

        self._init_weights()  # 初始化网络权重

    def _init_weights(self):
        # 隐藏层初始化
        self.W1 = np.random.randn(1, self.hidden_units) * np.sqrt(2.0 / 1)  # 输入维度为1
        self.b1 = np.zeros(self.hidden_units)  # 偏置初始化为0

        # 输出层初始化
        self.W2 = np.random.randn(self.hidden_units, 1) * np.sqrt(2.0 / self.hidden_units)
        self.b2 = np.zeros(1)  # 输出为标量

    def _standardize(self, x):
        return (x - self.x_mean) / self.x_std

    def _forward(self, x, is_training=True):
        if is_training and self.noise_std > 0:
            x = x + np.random.normal(0, self.noise_std, size=x.shape)

        h1 = np.dot(x, self.W1) + self.b1  # 加权求和
        a1 = np.maximum(0, h1)  # ReLU激活

        y_pred = np.dot(a1, self.W2) + self.b2
        return y_pred, a1, h1

    def _backward(self, x, a1, h1, dy_pred):
        # 输出层梯度
        dW2 = np.dot(a1.T, dy_pred)
        db2 = np.sum(dy_pred, axis=0)
        # 隐藏层梯度
        da1 = np.dot(dy_pred, self.W2.T)
        dh1 = da1 * (h1 > 0)
        # 输入层梯度
        dW1 = np.dot(x.T, dh1)
        db1 = np.sum(dh1, axis=0)

        return dW1, db1, dW2, db2

    def _clip_gradients(self, dW1, db1, dW2, db2):
        # 计算梯度总范数
        grad_norm = np.sqrt(np.sum(dW1 ** 2) + np.sum(db1 ** 2)+ np.sum(dW2 ** 2) + np.sum(db2 ** 2))

        # 如果梯度超过阈值，进行缩放
        if grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / grad_norm
            dW1 *= scale
            db1 *= scale
            dW2 *= scale
            db2 *= scale
        return dW1, db1, dW2, db2

    def fit(self, x_train, y_train, x_test, y_test, epochs=10000, verbose=100):
        # 数据标准化预处理
        self.x_mean, self.x_std = x_train.mean(), x_train.std()
        x_train = self._standardize(x_train)
        x_test = self._standardize(x_test)

        # 记录训练过程
        self.train_loss = []
        self.test_loss = []

        # 训练循环
        for epoch in range(epochs):
            # 学习率衰减（每2000轮次衰减到原来的0.9倍）
            current_lr = self.learning_rate * (0.9  ** (epoch // 2000))

            # 前向传播
            y_pred, a1, h1 = self._forward(x_train)

            # 计算训练损失（均方误差）
            train_loss = np.mean((y_pred - y_train)  ** 2)
            self.train_loss.append(train_loss)

            # 测试集评估（关闭噪声）
            test_pred, _, _ = self._forward(x_test, is_training=False)
            test_loss = np.mean((test_pred - y_test)  ** 2)
            self.test_loss.append(test_loss)

            # 反向传播计算梯度
            dy_pred = 2 * (y_pred - y_train) / len(x_train)  # MSE梯度
            dW1, db1, dW2, db2 = self._backward(x_train, a1, h1, dy_pred)

            # 梯度裁剪
            dW1, db1, dW2, db2 = self._clip_gradients(dW1, db1, dW2, db2)

            # 动量更新计算
            self.vW1 = self.mu * self.vW1 - current_lr * dW1
            self.vb1 = self.mu * self.vb1 - current_lr * db1
            self.vW2 = self.mu * self.vW2 - current_lr * dW2
            self.vb2 = self.mu * self.vb2 - current_lr * db2

            # 参数更新
            self.W1 += self.vW1
            self.b1 += self.vb1
            self.W2 += self.vW2
            self.b2 += self.vb2

            # 打印训练信息
            if verbose and epoch % verbose == 0:
                print(
                    f'Epoch {epoch:5d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {current_lr:.5f}')

    def predict(self, x):
        x = self._standardize(x)  # 标准化输入
        y_pred, _, _ = self._forward(x, is_training=False)  # 关闭噪声
        return y_pred


if __name__ == "__main__":
    # 目标函数定义
    def f(x):
        """待逼近的目标函数：x + sin(x)"""
        return x + np.sin(x)
    # 生成数据集
    np.random.seed(42)
    x_train = np.random.uniform(-5, 5, (500, 1))  # 400个训练样本，范围[-5,5)
    y_train = f(x_train)
    x_test = np.random.uniform(-5, 5, (100, 1))  # 100个测试样本
    y_test = f(x_test)

    # 初始化并训练模型
    model = ReluNetwork(
        hidden_units=128,  # 隐藏层神经元数量
        learning_rate=0.01,  # 初始学习率
        mu=0.9,  # 动量系数
        noise_std=0.02,  # 输入噪声标准差
        max_grad_norm=1.0  # 最大梯度范数
    )
    model.fit(x_train, y_train, x_test, y_test, epochs=2000, verbose=200)

    # 可视化结果
    plt.figure(figsize=(12, 5))

    # 函数拟合情况可视化
    plt.subplot(1, 2, 1)
    x_plot = np.linspace(-5, 5, 500).reshape(-1, 1)  # 生成画图用数据
    y_true = f(x_plot)
    y_pred = model.predict(x_plot)

    plt.plot(x_plot, y_true, label='True Function', linewidth=2, color='darkblue')
    plt.plot(x_plot, y_pred, '--', label='Prediction', linewidth=2, color='orange')
    plt.scatter(x_test, y_test, s=30, marker='x', label='Test Samples', color='red')

    plt.title('Function Approximation')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    # 训练过程损失曲线
    plt.subplot(1, 2, 2)
    plt.semilogy(model.train_loss, label='Training Loss', color='blue')
    plt.semilogy(model.test_loss, label='Test Loss', color='red')
    plt.title('Training Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()