from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import random


# import sys
# sys.setrecursionlimit(1000000)

class Poiter:
    position = np.array([])
    isvisited = False
    cluster = -1

    def __init__(self, position):
        self.position = np.array(position)

    def dist2(self, pointer2):  # 计算距离的平方
        return np.sum(np.square(self.position - pointer2.position))


class DBSCAN:
    epspow = 0.0  # 密度半径的平方
    minpnum = 0

    unvisitednum = 0
    pointers = []
    clusters = []
    clustersnum = 0

    def __init__(self, eps, minpnum):
        self.epspow = eps ** 2
        self.minpnum = minpnum

    def isker(self, pointer):  # 判断是否为核心点
        innum = 0
        for vpointer in self.pointers:
            if (vpointer.dist2(pointer) <= self.epspow):
                innum += 1
                if (innum >= self.minpnum):
                    return True
        return False

    def subcluster(self, pointer):
        # 此处的递归并不能解决一开始就被标记为噪点的点，所以在后面需要单独处理
        for vpointer in self.pointers:
            if (vpointer.isvisited == False and vpointer.dist2(pointer) <= self.epspow):
                # 未迭代到的，且距离小于eps的点，则发展为下线
                vpointer.cluster = pointer.cluster
                vpointer.isvisited = True
                self.unvisitednum -= 1
                if (self.unvisitednum == 0):  # 结束递归
                    return
                if (self.isker(vpointer)):
                    self.subcluster(vpointer)  # 递归“发展下线”

    def fit(self, pointers):
        self.unvisitednum = len(pointers)  # 记录未迭代到的点数
        for pointer in pointers:  # 将点转化为类点
            self.pointers.append(Poiter(pointer))

        while (1):
            for vpointer in self.pointers:
                if (vpointer.isvisited == False):
                    if (self.isker(vpointer)):

                        vpointer.isvisited = True
                        self.unvisitednum -= 1

                        self.clusters.append(self.clustersnum)
                        vpointer.cluster = self.clustersnum
                        self.subcluster(vpointer)  # 发展下线
                        self.clustersnum += 1

                    else:  # 噪声点类为-1
                        vpointer.isvisited = True
                        self.unvisitednum -= 1
                        vpointer.cluster = -1

            if (self.unvisitednum == 0):
                break
        # 最后处理边界上被误分的"伪噪点"
        for vpointer in self.pointers:
            if (self.isker(vpointer)):
                for vvpointer in self.pointers:
                    if (vvpointer.dist2(vpointer) <= self.epspow):
                        vvpointer.cluster = vpointer.cluster

    def predict(self, pointers):
        y = []
        for vpointer in pointers:
            for vvpointer in self.pointers:
                if ((vvpointer.position == vpointer).all()):
                    y.append(vvpointer.cluster)
                    break
        y = np.array(y)
        return y


if __name__ == '__main__':
    sampnum = 300
    random_state = random.randint(0, 100)

    x, _ = make_blobs(centers=6, n_samples=sampnum, random_state=random_state)
    x = x[:, 0:2]

    xadd = np.array([-5, -6])
    x = np.insert(x, 0, xadd, axis=0)

    # x = np.array([[2, 1],
    #               [0, 1],
    #               [0, 0],
    #               [1, 0],
    #               [1, 1]])

    dbscan = DBSCAN(1, 5)
    dbscan.fit(x)
    y = dbscan.predict(x)

    for i in dbscan.pointers:
        print(i.cluster, end=' ')
    # print(np.unique(y))

    plt.figure()

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan','yellow')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx))
    plt.show()
