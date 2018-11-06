import numpy as np
import matplotlib.pylab as plot


class Logistic:
    '''
    特征系数θ
    '''
    theta = np.zeros((1, 1))
    '''
    梯度上升循环次数
    '''
    cycleNum = 10000
    '''
    特征向量X、标记向量y
    '''
    X = np.zeros((1, 1))
    Y = np.zeros((1, 1))
    alpha = 1

    def z(self, X):  # z函数，决定z函数的形式 注：这里的X是向量不是矩阵
        return X.dot(self.theta.transpose())

    def h(self, x):
        return 1.0 / (1 + np.exp(-self.z(x)))

# 优化函数
    def fit(self, X, Y):
        cx = np.ones((X.shape[0], 1))
        self.X = np.c_[cx, X]#增加常数1的列
        self.Y = Y
        self.theta = np.random.random((1, self.X.shape[1]))  # 由于theta使用random函数导致其为二维而不是一维。
        i = 0
        j = 0
        while j < self.cycleNum:
            dtheta = np.zeros((1, self.X.shape[1]))
            # print(self.theta)
            # print(self.theta[0][0] / self.theta[0][1])
            while i <= self.Y.shape[0] - 1:
                dtheta += (self.Y[i] - self.h(self.X[i])) * self.X[i]
                i += 1
            i = 0  # 初始化i

            self.theta = self.theta + self.alpha * dtheta
            j += 1

    def predict(self, vX):
        output = self.h(vX)
        if output < 0.5:
            return 0
        else:
            return 1


if __name__ == '__main__':

    lineSplit = []
    x_train = []
    y_train = []
    with open("testSet-LR.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            lineSplit = (line.strip().split())
            x_train.append([float(lineSplit[0]), float(lineSplit[1])])
            y_train.append([int(lineSplit[2])])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    logis = Logistic()
    logis.alpha = 100
    logis.cycleNum = 30000

    logis.fit(x_train, y_train)

    xop = []
    yop = []
    xpe = []
    ype = []
    i = 0
    while i <= x_train.shape[0] - 1:
        if y_train[i] == 1:
            xop.append(x_train[i][0])
            yop.append(x_train[i][1])
        else:
            xpe.append(x_train[i][0])
            ype.append(x_train[i][1])
        i += 1

    fig = plot.figure()
    plot.scatter(xop, yop, color="red")
    plot.scatter(xpe, ype, color="blue")
    plot.xlim((-10, 20))
    plot.ylim((-10, 20))

    X = np.linspace(-10, 10, 30)
    Y = -X * logis.theta[0][1] / logis.theta[0][2] - logis.theta[0][0] / logis.theta[0][2]
    plot.plot(X, Y)

    plot.show()
    fig.savefig('lr.jpg')
