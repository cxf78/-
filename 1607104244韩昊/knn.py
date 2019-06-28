import numpy as np
from DSVC.data_utils import load_CIFAR10
import random
import matplotlib.pyplot as plt
from DSVC.k_nearest_neighbor import KNearestNeighbor

#导入数据
cifar10_dir = 'DSVC/datasets/cifar-10-batches-py' 
x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir,6)
#数据采样
num_training=1000
mask=range(num_training)
x_train=x_train[mask]
y_train=y_train[mask]
num_test=200
mask=range(num_test)
x_test=x_test[mask]
y_test=y_test[mask]
#将图片数据重新整理
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
#使用KNN进行训练
knn = KNearestNeighbor()
knn.train(x_train,y_train)
dists = knn.getdist(x_test)
m=1
y_pred =knn.predict(dists,k=m)
accuracy = float(np.sum(y_pred == y_test)) / num_test
print ('k=%d; accuracy: %f' % (m, accuracy))
#交叉验证
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

x_train_folds = []
y_train_folds = []
x_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] =[]
for k in k_choices:
    for j in range(num_folds):
        x_train_cv = np.vstack(x_train_folds[0:j]+x_train_folds[j+1:])
        x_test_cv = x_train_folds[j]  
        y_train_cv = np.hstack(y_train_folds[0:j]+y_train_folds[j+1:])
        y_test_cv = y_train_folds[j]
        
        knn.train(x_train_cv,y_train_cv)
        dists_cv = knn.getdist(x_test_cv)
        y_test_pred = knn.predict(dists_cv,k)
        num_correct = np.sum(y_test_pred == y_test_cv)
        acc = float(num_correct)/ num_test
        k_to_accuracies[k].append(acc)
print('交叉验证：')
# print (k_to_accuracies)
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print ('k = %d, accuracy = %f' % (k, accuracy))