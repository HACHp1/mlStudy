import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 二次回归

def f(x):
    return 2 * x * x + 1


x = np.linspace(0, 30, 30)  # 三十个数据点的x坐标
x2 = np.random.randint(-3, 3, (30))  # 加入随机化震荡
y = f(x + x2)  # 数据点的纵坐标

epoch = 10  # 整个数据集循环使用多少次

plt.figure()  # 画点图
plt.scatter(x, y)

w = tf.Variable(1, dtype=tf.float32, name='weight')  # 参数1
b = tf.Variable(1, dtype=tf.float32, name='bias')  # 参数2
xh = tf.placeholder(tf.float32)  # 因为使用随机梯度下降，加上placeholder
yh = tf.placeholder(tf.float32)
predict = tf.multiply(w, xh * xh) + b  # 预测值

cost = tf.reduce_mean((tf.square(predict - yh)) / 2)  # 最小二乘法代价函数
optimizer = tf.train.AdamOptimizer(0.01)  # 使用ADAM优化，学习率0.01
train_step = optimizer.minimize(cost)  # 一个训练步骤

init = tf.initialize_all_variables()  # 初始化变量的operation

with tf.Session() as sess:
    sess.run(init)  # 初始化
    for j in range(epoch):  # 进行epoch次大循环
        for i in range(30):  # 对每个数据点的遍历
            sess.run(train_step, {xh: x[i], yh: y[i]})  # 塞入一个数据点
            print(sess.run(cost, {xh: x, yh: y}))  # 查看代价函数的值

    # 训练结束
    wshow = sess.run(w)  # 查看参数1
    bshow = sess.run(b)  # 查看参数2
    print(wshow, ' ', bshow)  # 输出
    # 对训练完的函数进行画图
    xplot = np.linspace(-1, 31, 30)
    plt.plot(xplot, wshow * xplot * xplot + bshow, c="red")
    plt.show()
