#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from os import listdir
font=FontProperties(fname='/Library/Fonts/Songti.ttc')

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#计算dataSet里总共有多少样本
    diffMat = tile(inX, (dataSetSize,1)) - dataSet#利用tile函数扩展预测数据，与每一个训练样本求距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5#求曼哈顿距离
    sortedDistIndicies = distances.argsort()#argsort()函数用法见下
    classCount={} #定义一个字典，用于储存K个最近点对应的分类以及出现的频次
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #以下代码将不同labels的出现频次由大到小排列，输出次数最多的类别
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#########
def file2matrix():
    dataMat = []
    labelMat = []  # dataMat为X输入数据，labelMat为输出0，1数据
    fr = open('/Users/cailei/Cai_Lei/AI/MachineLearning_data/Ch02/datingTestSet2.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # 移除头尾指定字符，并将数据分割为不同部分
        dataMat.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2])])  # 令X0为第一列数据，X1为第二列数据，X2为第三列数据
        labelMat.append(int(lineArr[3]))  # 将第4列数据赋值为输出数据
    return dataMat, labelMat


####归一化特征值,将所有特征值归一化
def autoNorm(dataSet):
    NPdataSet = np.array(dataSet)
    minVals = NPdataSet.min(0)
    maxVals = NPdataSet.max(0)
    ranges = np.array(maxVals - minVals)
    normDataSet = np.zeros(shape(dataSet))
    m = NPdataMat.shape[0]
    normDataSet = NPdataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals,maxVals

###测试分类器准确率代码
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels=file2matrix()
    NPdatingDataMat=np.array(datingDataMat)
    NPdatingLabelMat=np.array(datingLabels)
    normDataSet,ranges,minVals=autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult=classify0(NPdatingDataMat[i,:],NPdatingDataMat[numTestVecs:m,:],NPdatingLabelMat,3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult,datingLabels[i])
        if (classifierResult!= datingLabels[i]): errorCount+=1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range (32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('/Users/cailei/Cai_Lei/AI/MachineLearning_data/Ch02/digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('/Users/cailei/Cai_Lei/AI/MachineLearning_data/Ch02/digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('/Users/cailei/Cai_Lei/AI/MachineLearning_data/Ch02/digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/Users/cailei/Cai_Lei/AI/MachineLearning_data/Ch02/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


if __name__ == '__main__':
    dataMat,labelMat=file2matrix()
    NPdataMat=np.array(dataMat)
    NPlabelMat=np.array(labelMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(NPdataMat[:,0],NPdataMat[:,1],15.0*array(NPlabelMat),15.0*array(NPlabelMat))
    plt.title(u'二三列数据plot', fontproperties=font)
    plt.xlabel(u'玩视频游戏所耗时间百分比', fontproperties=font)
    plt.ylabel(u'每周消费的冰淇淋公升数', fontproperties=font)
    #plt.show()
    #datingClassTest()
    handwritingClassTest()