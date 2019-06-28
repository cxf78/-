import numpy as np
import random
from Knn import Knn

# 用于计算准确度
def caculate_accuracy(predictRsl,origiRsl):
    dominator=len(predictRsl)
    numerator=0
    for i in range(dominator):
        if predictRsl[i] == origiRsl[i]:
            numerator+=1
    accuracy=numerator/dominator
    return accuracy

# 交叉验证
def cross_validation(origiData,origiLabel,splitNum):
    lastIndex=0
    offset=int(len(origiData)/splitNum)
    accurateRateSum=0
    for i in range(1,splitNum + 1):
        # 切割数据集 分为splitNum-1个训练集和1个测试集
        tempData=np.split(origiData,(lastIndex,i*offset+1))
        tempLabel=np.split(origiLabel,(lastIndex,i*offset+1))

        testData=tempData[1]
        testLabel=tempLabel[1]
        trainData=np.concatenate([tempData[0],tempData[2]])
        trainLabel=np.concatenate([tempLabel[0],tempLabel[2]])
        # 迭代区间的开始索引值
        lastIndex=i*offset
        
        # 使用knn进行预测
        knn=Knn(trainData,trainLabel,10)
        predictRsl=knn.predict(testData)
        accuracy=caculate_accuracy(predictRsl,testLabel)
        accurateRateSum+=accuracy
    # 返回预测结果的平均值
    return accurateRateSum/splitNum

        