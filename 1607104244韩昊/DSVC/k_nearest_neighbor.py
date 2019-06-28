import numpy as np
from collections import Counter
class KNearestNeighbor(object):
    def __init__(self):
        pass
    def train(self , x, y):   
        self.x_train=x
        self.y_train=y
    def predicts(self, x, k=1):
        dists = self.getdist(x)
        return self.predict(dists, k=k)
  
    def getdist(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train): 
                dists[i,j] = np.linalg.norm(self.x_train[j,:]-x[i,:])
        return dists
    def predict(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
         labels = self.y_train[np.argsort(dists[i,:])].flatten()
         closest_y = labels[0:k]
         c = Counter(closest_y)
         y_pred[i] = c.most_common(1)[0][0]
        return y_pred