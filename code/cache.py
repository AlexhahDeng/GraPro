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
np.set_printoptions(suppress=True)
from func import *

dataPath=os.path.abspath(os.path.dirname(os.getcwd())) + "\data\\" 

# 边缘节点容量
EDGECAPACITY = 2000

# 视频最大size（主要用于随机生成大小
VIDEOMAXSIZE = 200

# 获取ahp权重
WEIGHT = ahp()

POPULATION_SIZE = 10 # 初始化种群大小
GENERATION_TIME = 200 # 最大迭代次数
MAX_SCORE = 5 # FIXME 咱也不知道，只能先随便整一个
CROSS_RATE = 0.5 # 交叉概率
MUTATE_RATE = 0.01 # 变异概率

def overallRating(video):
    '''
    func:
        1、根据流行度对于视频打分
    return：
        分数
    '''
    score = 0.0
    for i in range(len(WEIGHT)):
        score += WEIGHT[i] * video[i]

    return score

def generateData():
    '''
    func：在万般无奈下只能自己编造数据了
    return：array col1：movieId col2:size col3:scores
    '''
    # col1：电影点击数 col2：电影平均评分 col3：电影评论数 
    listMovie = [[0.0 for col in range(3)] for row in range(MAXNUM)] 
    listMovie = np.asarray(listMovie)

    listMovie = getClicksandAvgScores(listMovie)
    listMovie = getCommentsNum(listMovie)

    # 搞一些随机数据
    edgeMovies = []
    edgeSize = EDGECAPACITY
    colMax = np.max(listMovie, axis = 0) # 获取每行数据最大值

    for i in range(len(listMovie[0,:])): # 归一化处理数据
        listMovie[:,i] = listMovie[:,i] / colMax[i]

    for i in range(len(listMovie)):
        if(listMovie[i,0] == 0):
            continue

        else:
            # 一条视频信息
            videoInfo = []
            videoInfo.append(i) # id

            # 随机生成视频大小
            videoSize = random.randint(5,VIDEOMAXSIZE)

            while(edgeSize - videoSize < 0): # 随机化大小不合适
                videoSize = random.randint(5,VIDEOMAXSIZE)

            videoInfo.append(videoSize) # 大小
            videoInfo.append(overallRating(listMovie[i])) # 流行度分数

            edgeMovies.append(videoInfo)
            edgeSize -= videoSize

            if(edgeSize < 5):
                break

    print("随机数据生成！\n")
    
    return np.asarray(edgeMovies)

def generatePopulation(group_size, gene_len):
    '''
    func：
        1、初始化种群，大小可以改变（不保证每个个体都不一样，不过一毛一样的概率较低）
        2、传入参数为边缘结点目前电影个数
        3、每个序列代表一种替换方案，但并不保证替换符合要求（比如大小超过了）
    return：
        list，种群所有个体
    '''
    group = []
    for i in range(group_size):
        individual = []
        for j in range(gene_len):
            individual.append(random.randint(0,1)) # 随机化染色体序列咯
        
        # 为了变成好用的东西
        tmp = []
        tmp.append(individual)
        tmp.append([])
        group.append(tmp)

    print("种群初始化……\n")

    return group


def fitness(edgeMovies, leftRoom, individual):
    # 首先判断大小是不是超过限制，若是，则直接适应度为负数，毙掉
    totalSize = 0
    scores = 0.0

    for j in range(len(individual)):
        if(individual[j] == 1):
            totalSize += edgeMovies[j][1]
            scores += edgeMovies[j][2]
        else:
            continue
    
    if(totalSize > leftRoom):
        return -1
    else:
        return scores


def random_pick(some_list,probabilities):
    x = random.uniform(0,1)
    cumulative_probability=0.0
    count = 0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
        count += 1
    return count
# BUG 魔改了一下

def cross(group, leftRoom, edgeMovies):

    # FIXME 先用随机选择的方法⑧
    childList = []
    group = np.asarray(group)

    sum = np.sum(group[:,1])
    pos = group[:,1]/sum

    for i in range(len(pos)):
        pos[i] = round(pos[i], 3)
    
    # 实在是没办法保证概率之和为1，只能随机加减了
    pos[random.randint(0, len(pos)-1)] -= (np.sum(pos) - 1)

    index = random_pick(group[:,1], pos)
    # print(index)
    # TODO 能够按轮盘赌办法随机选择个体了




    return 

def GA(edgeMovies, leftRoom):
    '''
    func：
        1、在剩余空间的限制下，从文件中选择丢弃哪些文件
        2、保证剩余的文件能达到最优的命中率
    return：
        最佳video组合
    '''

    # group结构[[[染色体序列]，[fitness]],……]
    Group = generatePopulation(POPULATION_SIZE ,len(edgeMovies))

    print("种群大小     " + str(POPULATION_SIZE) + "\n" + "染色体长度   " + str(len(edgeMovies)) + "\n")
    print("剩余空间大小     " + str(leftRoom) + "\n")

    # 计算适应度
    for i in range(len(Group)):
        individual = Group[i][0]
        scores = fitness(edgeMovies, leftRoom, individual)

        # 分数不合要求直接毙掉
        if ( scores == -1):
            Group.pop(i)
        else:
            Group[i][1] = scores

    cross(Group, leftRoom, edgeMovies)
    # for i in range(GENERATION_TIME):

    #     # 交叉


    #     # 变异




    


            





def main():

    edgeMovies = generateData()

    # 随机生成视频请求，目前还是限制一下大小，从简
    videoRequest = random.randint(5, VIDEOMAXSIZE)
    leftRoom = np.sum(edgeMovies[:,1]) - videoRequest

    # 遗传算法
    GA(edgeMovies, leftRoom)




main()
