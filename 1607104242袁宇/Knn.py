import numpy as np
import math





class Knn:
    def __init__(self,trainData,trainLabel,k):
        self.trainData=trainData
        self.trainLabel=trainLabel
        self.k=k

    def caculate_distance_loop_version(self,target,contrast):
        dis=0
        for i in range(0,32):
            for j in range(0,32):
                PixelBias=0
                for k in range(0,3):
                    tarElement=target[i][j][k]
                    conElement=contrast[i][j][k]
                    PixelBias+=abs(tarElement-conElement)/3.0
                dis+=math.sqrt(PixelBias)
        dis=math.sqrt(dis)
        return dis

    def caculate_distance_matrix_version(self,target,contrast):
        temp=np.abs(target-contrast)
        dis=math.sqrt(np.sum(temp)/3)
        return dis

    def Contrast_All_TrainData(self,testElement):
        elementDic=[]
        elementLabel=[]
        for i in range(len(self.trainData)):
            dis=self.caculate_distance_matrix_version(testElement,self.trainData[i])
            elementDic.append(dis)
            elementLabel.append(self.trainLabel[i])
        return elementDic,elementLabel

    def predict(self,testData):
        rsl=[]
        for i in range(len(testData)):
            elementDic,elementLabel=self.Contrast_All_TrainData(testData[i])
            sortIndex=np.argsort(elementDic)[:self.k]
            elementLabel=np.array(elementLabel)[sortIndex]
            predictLabel=np.argmax(np.bincount(elementLabel))
            rsl.append(predictLabel)
        return rsl
        