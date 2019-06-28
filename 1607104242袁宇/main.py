from LoadData import loadTestData
from LoadData import loadTrainData
from Knn import Knn
from CrossValidation import cross_validation
import numpy as np
import math


trainData,trainLabel=loadTrainData("cifar-10-batches-py/")
print(np.shape(trainData),np.shape(trainLabel))

testData,testLabel=loadTestData("cifar-10-batches-py/")
print(np.shape(testData),np.shape(testLabel))

trainData=trainData[:100]
trainLabel=trainLabel[:100]
testData=testData[:10]
testLabel=testLabel[:10]

accuracy = cross_validation(trainData,trainLabel,4)
print(accuracy)
