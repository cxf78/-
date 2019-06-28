import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import os

def loadBatch(fileName):
    f=open(fileName,"rb")
    dataInterval=pk.load(f,encoding='latin1')
    data=dataInterval['data']
    data=data.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
    label=dataInterval['labels']
    label=np.array(label)
    return data,label

def loadTrainData(fileName):
    data=[]
    label=[]
    for i in range(1,6):
        filePath=os.path.join(fileName+"data_batch_%s"%i)
        d,l=loadBatch(filePath)
        data.append(d)
        label.append(l)
    data=np.concatenate(data)
    label=np.concatenate(label)
    return data,label

def loadTestData(fileName):
    filePath=os.path.join(fileName+"test_batch")
    data,label=loadBatch(filePath)
    return data,label



