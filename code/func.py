import csv
import os
import sys
import heapq
import json
import math
import random
import pandas as pd
import numpy as np
import pandas as pd
import datetime
np.set_printoptions(suppress=True)

# 数据路径
dataPath=os.path.abspath(os.path.dirname(os.getcwd())) + "\data\\" 
# 结果路径
resultPath = os.path.abspath(os.path.dirname(os.getcwd())) + "\\result\\" 

# KEYPOINT 电影数
MAXNUM = 1000000

# KEYPOINT 电影数
MAXNUM = 1000000

# ahp一致性检验参数，阶为3的情况，如果后期要加还要根据情况变化
RI = 0.58
def ahp():
    '''
    func:利用ahp层次分析法获取权重
    return：numpy array
    '''

    # 创建成对矩阵
    arr = np.array([[1,5,3],
                    [1/5,1,1/3],
                    [1/3,3,1]])

    col_sum = arr.sum(axis = 0) # 列求和

    # 矩阵归一化    
    [rows, cols] = arr.shape
    brr = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            brr[i,j] = arr[i,j] / col_sum[j]

    row_sum = brr.sum(axis = 0) # 行求和
    w = [0 for i in range(3)]

    # 归一化权重
    for i in range(len(row_sum)):
        w[i] = row_sum[i] / row_sum.sum()

    # 检验一致性(特征值，特征向量)
    evalue,fevector=np.linalg.eig(arr)

    evalue_max = np.max(evalue) # 最大特征值

    CI = (evalue_max - rows) / (rows - 1)

    CR = CI/RI

    # if(CR < 0.1):
    #     print("CR:" + str(CR) +"    通过一致性检验\n")
    # else:
    #     print("未通过一致性检验，请重新设计对比矩阵\n")

    # print("权重因子为"+str(w))
    return w # 权重array


def readCSV(filePath):
    '''
    func:读取csv的内容
    '''
    file=open(filePath,'r',encoding="gbk")# 读取以utf-8
    try:
        context = file.read() # 读取成str
        list_result=context.split("\n")#  以回车符\n分割成单独的行

        #每一行的各个元素是以【,】分割的，因此可以
        length=len(list_result)

        for i in range(1, length): 
            list_result[i]=list_result[i].split(",")
        
        list_result.remove(list_result[0]) # 删除第一行

        return list_result

    except Exception:
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    
    finally:
        file.close();# 操作完成一定要关闭


def toCSV(filename, name, list1, hasindex = False):
    '''
    para:filename,name=[], list1=[]
    func:把listMovie数据写入 csv（是list类型的哦，当心别传入array
    '''
    filename = resultPath + filename + ".csv"
    test = pd.DataFrame(columns = name, data = list1)
    test.to_csv(filename,index = hasindex, encoding = 'gbk')

    return 


def getClicksandAvgScores(listMovie):
    '''
    func:获取点击量和平均评分，结果计入listMovie数组
    '''
    filename = dataPath + "ratings.csv"
    list1 = readCSV(filename)

    for i in range(len(list1)): #userId,movieId,rating,timestamp 第一行是header
        movieNum = int(list1[i][1]) # movieid
        
        listMovie[movieNum][0] = listMovie[movieNum][0] + 1 # count clicks
        listMovie[movieNum][1] = float(list1[i][2]) + listMovie[movieNum][1] # count scores

    for i in range(len(listMovie)):
        if(listMovie[i][0] != 0):
            listMovie[i][1] = listMovie[i][1] / listMovie[i][0]

    # print("获取点击量和平均评分……done\n")
    return listMovie


def getCommentsNum(listMovie):
    '''
    func: 获取评论数，结果直接计入listMovie数组
    '''

    filename = dataPath + "tags.csv"
    list1 = readCSV(filename)

    for i in range(len(list1)): #userId,movieId,tag,timestamp
        movieNum = int(list1[i][1])

        listMovie[movieNum][2] = listMovie[movieNum][2] + 1

    # print("获取评论数……done\n")
    return listMovie


def generateEdgeUser(edgeUserNum):
    '''
    func: given edge user number, and generate user randomly
    '''

    # 获取随机生成的边缘用户集
    allUser = np.asarray(readCSV(dataPath + "ratings.csv"))
    userId = allUser[:,0]
    edgeUserId = random.sample(list(userId), edgeUserNum)
    


    return edgeUserId


def edgeUser(edgeUserId):
    '''
    func:
        1、针对用户观影记录，获取电影list，index是id，然后有评论数，观看量和平均评分
        2、根据时间戳，分成前15年训练movieList，后7年为测试testMovieList
    points：
        1、可以改动边缘用户数量来看影响
        2、可以改动时间戳，来改变训练和测试的数据量
    return: 边缘用户集array，测试电影array
    '''

    # 读取打分和标签文件
    rating = np.asarray(readCSV(dataPath + "ratings.csv"))
    tag = np.asarray(readCSV(dataPath + "tags.csv"))

    # 获取边缘用户观影记录
    edgeUserMovieList = [[0.0 for col in range(4)] for row in range(MAXNUM)] 
    testMovieList = [] # KEYPOINT 只存11年以后的电影id

    # 数据集时间跨度：1996/0529--20180924（22年）取前15年训练好了
    d_start = datetime.datetime(1970,1,1)
    d_end = datetime.datetime(2011,1,1)
    train_seconds = (datetime.datetime(2012,1,1) - d_start).days * 24 * 3600 # 测试集选1年

    seconds = (d_end - d_start).days * 24 * 3600 # 计算秒数，时间戳大于这个就纳入训练

    for i in range(len(edgeUserId)):    
        userid = edgeUserId[i]
        # print(userid)
        MovieList = rating[np.where(rating[:,0] == userid)]
        TagList = tag[np.where(tag[:,0] == userid)]

        # 针对每个用户看过的电影遍历
        for j in range(len(MovieList)):
            timestamp = int(MovieList[j][3])
            movieid = int(MovieList[j][1])

            if(timestamp > seconds and timestamp < train_seconds): # 训练集
                if(movieid not in testMovieList):
                    testMovieList.append(movieid)
            else: 
                edgeUserMovieList[movieid][0] += 1 # 点击量++
                edgeUserMovieList[movieid][1] += float(MovieList[j][2]) # 评分++

        # 针对每个用户做过标记的电影遍历
        for j in range(len(TagList)):
            movieid = int(TagList[j][1])
            edgeUserMovieList[movieid][2] += 1 # 评论数++

    # 弄成平均分
    for i in range(len(edgeUserMovieList)):
        if(edgeUserMovieList[i][0] == 0):
            continue
        else:
            edgeUserMovieList[i][1] = edgeUserMovieList[i][1] / edgeUserMovieList[i][0]


    # print(np.max(np.asarray(edgeUserMovieList), axis = 0))
    # print("测试电影集…………" + str(testMovieList) + "/n")

    return np.asarray(edgeUserMovieList), np.asarray(testMovieList)


# 后期备用
# def main():
#     # 获取权重
#     ahpWei = ahp()

#     # 填满电影数据矩阵
#     getClicksandAvgScores(listMovie)
#     getCommentsNum(listMovie)

#     # 将最热门的电影整合，模拟某用户观影记录
#     targetUser = getHotMovies(ahpWei, listMovie)

#     # 获取每个用户的观影记录
#     userMovieList = getUserMovies()

#     # 获取推荐电影集
#     getRecommMovies(targetUser, userMovieList)