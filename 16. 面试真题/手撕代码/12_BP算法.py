import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(z, 0)


class Network(object):
    def __init__(self):
        super(Network, self).__init__()

        self.w1 = np.random.randn(3, 10)
        self.b1 = np.random.randn(10)
        self.w2 = np.random.randn(10, 1)
        self.b2 = np.random.randn(1)

        self.grads = []

    def forward(self, x, y=None):
        """
        前向过程执行
        :param x: numpy数组，形状为[batch_size, in_features]
        :param y: numpy数组，形状为[batch_size, out_targets]
        :return: (前向执行结果，损失函数值)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        z1 = np.dot(x, self.w1) + self.b1  # [N,10]
        z2 = sigmoid(z1)  # [N,10]
        # z2 = relu(z1)  # [N,10]
        y_pred = np.dot(z2, self.w2) + self.b2  # [N,1]

        loss = None
        if y is not None:
            y = np.asarray(y)
            # 计算损失
            loss = 0.5 * np.square(y_pred - y).sum()
            # 计算梯度值
            grad_y_pred = y_pred - y  # [N,1]
            grad_b2 = grad_y_pred.sum(axis=0)  # [1]
            grad_w2 = z2.T.dot(grad_y_pred)  # [10, 1]
            grad_z2 = grad_y_pred.dot(self.w2.T)  # [N,10]
            # relu提取求解
            # grad_z1 = grad_z2.copy()
            # grad_z1[z2 < 0] = 0  # relu输出为0的位置梯度为0
            grad_z1 = grad_z2 * z2 * (1 - z2)  # [N,10] # sigmoid梯度求解
            grad_w1 = x.T.dot(grad_z1)  # [3,10]
            grad_b1 = grad_z1.sum(axis=0)

            # 临时保存梯度
            self.grads = [grad_b1, grad_w1, grad_b2, grad_w2]

        return y_pred, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, lr=0.01):
        if len(self.grads) != 4:
            raise ValueError("请先调用forward计算梯度值")
        self.b1 = self.b1 - lr * self.grads[0]
        self.w1 = self.w1 - lr * self.grads[1]
        self.b2 = self.b2 - lr * self.grads[2]
        self.w2 = self.w2 - lr * self.grads[3]
        self.grads = []


if __name__ == '__main__':
    # 可参考： https://zhuanlan.zhihu.com/p/47051157
    np.random.seed(28)
    net = Network()
    dx = np.random.randn(5, 3)
    dy = dx[:, 0] ** 2 + dx[:, 1] ** 3 + dx[:, 2] * dx[:, 0] * dx[:, 1] + np.random.randn(5)
    dy = np.reshape(dy, (-1, 1))
    print(dy)
    xx = net(dx, dy)
    print(xx)
    print("=" * 10)
    for i in range(2000):
        xx = net(dx, dy)
        net.backward()
    print(xx)