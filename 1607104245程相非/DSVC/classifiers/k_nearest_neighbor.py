import numpy as np


class KNearestNeighbor(object):
    """knn分类器"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        训练分类器。这只是记住所有的训练数据。
        投入：
            -x：包含训练数据的形状（num_train，d）的Numpy阵列
                由每个维度的num_火车样本组成。
            -Y：包含训练标签的形状（num_train，）的Numpy阵列，
                其中Y[I]是x[I]的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        测试分类器。
        投入：
            -x：包含测试数据的形状(num_test,d)的Numpy数组
                由每个维度的num测试样品组成.
            - 最近的邻居投票给预测的标签的数目。
            -num_loop：确定是否使用for-loop来计算l2距离
                        在训练集和测试集之间
        返回：
            -pred_Y：预测输出Y
        """
        # 计算测试x与x之间的距离
        if num_loops == 0:
            # 无循环，矢量化
            dists = self.cal_dists_no_loop(X)
        elif num_loops == 1:
            # 一个循环，半向量
            dists = self.cal_dists_one_loop(X)
        elif num_loops == 2:
            # 两个循环，没有矢量化
            dists = self.cal_dists_two_loop(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        # 预测标签
        num_test = X.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dists_k_min = np.argsort(dists[i])[:k]  # 最接近的K距离loc
            close_y = self.y_train[dists_k_min]  # 最接近的K距离，所有标签
            y_pred[i] = np.argmax(np.bincount(close_y))  # [0,3,1,3,3,1] -> 3　as y_pred[i]

        return y_pred

    def cal_dists_no_loop(self, X):
        """
        计算无循环的距离
        输入：
            -x：包含测试数据的形状(num_test,d)的Numpy数组
                由每个维度的num测试样品组成.
        返回：
            -dists：测试x和训练x之间的距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
        d1 = np.multiply(np.dot(X, self.X_train.T), -2)  # shape (num_test, num_train)
        d2 = np.sum(np.square(X), axis=1, keepdims=True)  # shape (num_test, 1)
        d3 = np.sum(np.square(self.X_train), axis=1)  # shape (1, num_train)
        dists = np.sqrt(d1 + d2 + d3)

        return dists

    def cal_dists_one_loop(self, X):
        """
        用一个循环计算距离
        输入：
            -x：包含测试数据的形状(num_test,d)的Numpy数组
                由每个维度的num测试样品组成.
        返回：
            -dists：测试x和训练x之间的距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis=1))

        return dists

    def cal_dists_two_loop(self, X):
        """

        用两个循环计算距离
        输入：
            -x：包含测试数据的形状(num_test,d)的Numpy数组
                由每个维度的num测试样品组成.
        返回：
            -dists：测试x和训练x之间的距离
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

        return dists