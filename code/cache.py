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

GENERATION_TIME = 200 # 最大迭代次数
MAX_SCORE = 5 # FIXME 咱也不知道，只能先随便整一个

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

def generateData(edgecapacity):
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
    edgeSize = edgecapacity
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


def cross(group, leftRoom, edgeMovies, crosstime, populationsize):

    # KEYPOINT 轮盘赌的方法⑧
    # childList = [] 暂且是用不上
    groupArray = np.asarray(group)

    sum = np.sum(groupArray[:,1])
    pos = groupArray[:,1]/sum

    for i in range(len(pos)):
        pos[i] = round(pos[i], 3)
    
    # 实在是没办法保证概率之和为1，只能随机加减了
    pos[random.randint(0, len(pos)-1)] -= (np.sum(pos) - 1)

    # print(str(group) + "???")
    # 开始交配~（怪怪的、次数自定
    for i in range(crosstime):

        # 随机选择个体，不能自交哦
        mo = random_pick(groupArray[:,1], pos)
        fa = random_pick(groupArray[:,1], pos)

        while(fa == mo):
            fa = random_pick(groupArray[:,1], pos)
        
        # 还是随机单点交叉把
        location = random.randint(0, len(pos) - 1)

        mo = groupArray[mo][0]
        mo_part = mo[:location]

        fa = groupArray[fa][0]
        fa_part = fa[:location]

        child1 = np.append(mo_part, fa[location:]) # mo_part.append(fa[location:])
        child2 = np.append(fa_part, mo[location:]) # fa_part.append(mo[location:])

        # 计算后代的适应度
        scores1 = fitness(edgeMovies, leftRoom, child1)
        scores2 = fitness(edgeMovies, leftRoom, child2)


        if(scores1 != -1):
            tmp = []
            tmp.append(list(child1))
            tmp.append(scores1)
            group.append(tmp)
        
        if(scores2 != -1):
            tmp = []
            tmp.append(list(child2))
            tmp.append(scores2)
            group.append(tmp)

    # 排序毙掉最弱的哦
    scores = list(np.asarray(group)[:,1])

    # 获取分数最高的n个
    top_n = list(map(scores.index, heapq.nlargest(populationsize, scores)))
    new_group = []

    for i in range(len(top_n)):
        new_group.append(group[top_n[i]])

    return new_group


def mutate(group, leftRoom, edgeMovies, mutatetime, populationsize):

    # 单点变异把
    for i in range(mutatetime):
        # 先随机选择变异个体把
        mutate_index = random.randint(0, len(group) - 1)
        individual = list(group[mutate_index][0]) # KEYPOINT 这是不用list的话，就把parent搭进去了

        # 选择变异位置
        mutate_location = random.randint(0, len(individual) - 1)

        if(individual[mutate_location] == 1):
            individual[mutate_location] == 0
        else:
            individual[mutate_location] == 1

        scores = fitness(edgeMovies, leftRoom, individual)

        if(scores != -1):
            tmp = []
            tmp.append(individual)
            tmp.append(scores)
            group.append(tmp)

    # 排序毙掉最弱的哦
    scores = list(np.asarray(group)[:,1])

    # 获取分数最高的n个
    top_n = list(map(scores.index, heapq.nlargest(populationsize, scores)))
    new_group = []

    for i in range(len(top_n)):
        new_group.append(group[top_n[i]])

    return new_group


def GA(edgeMovies, leftRoom, population_size, cross_time, mutate_time, generation_time):
    '''
    func：
        1、在剩余空间的限制下，从文件中选择丢弃哪些文件
        2、保证剩余的文件能达到最优的命中率
    return：
        最佳video组合
    '''

    # group结构[[[染色体序列]，[fitness]],……]
    Group = generatePopulation(population_size ,len(edgeMovies))

    print("种群大小     " + str(population_size) + "\n" + "染色体长度   " + str(len(edgeMovies)) + "\n")
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

    test_group = list(Group)
    for i in range(generation_time):

        # print("initial highest" + "     " + str(np.max(np.asarray(Group)[:,1])))
        # print(Group)
        # 交叉
        cross_group = cross(test_group, leftRoom, edgeMovies, cross_time, population_size)

        # print("after cross highest" + "     " + str(np.max(np.asarray(cross_group)[:,1])))
        # print(new_group)
        # 变异
        mutate_group = mutate(cross_group, leftRoom, edgeMovies, mutate_time, population_size)
        # print("after mutate highest" + "     " + str(np.max(np.asarray(mutate_group)[:,1])))

        test_group = mutate_group

    print("initial highest" + "     " + str(np.max(np.asarray(Group)[:,1])))
    print("final highest" + "        " + str(np.max(np.asarray(test_group)[:,1])))

        
    # test_group = list(Group)
    # for i in range(2000):

    #     # print("initial highest" + "     " + str(np.max(np.asarray(Group)[:,1])))
    #     # print(Group)
    #     # 交叉
    #     cross_group = cross(test_group, leftRoom, edgeMovies)

    #     # print("after cross highest" + "     " + str(np.max(np.asarray(cross_group)[:,1])))
    #     # print(new_group)
    #     # 变异
    #     mutate_group = mutate(cross_group, leftRoom, edgeMovies)
    #     # print("after mutate highest" + "     " + str(np.max(np.asarray(mutate_group)[:,1])))

    #     test_group = mutate_group

    # print("initial highest" + "     " + str(np.max(np.asarray(Group)[:,1])))
    # print("final highest" + "        " + str(np.max(np.asarray(test_group)[:,1])))




def dropMinFitness(edgeMovies, leftRoom):
    '''
    func：直接按照fitness来pass掉最低的一些视频，直到容量满足要求
    return：直接返回最后结果的流行度
    '''
    # 按照fitness升序排列，得到index的值
    movieIndex = edgeMovies[:,2].argsort()

    # 边缘节点初始容量
    initialSize = np.sum(edgeMovies[:,1])

    for i in range(len(movieIndex)):
        currIndex = movieIndex[i]
        initialSize -= edgeMovies[currIndex][1]
        edgeMovies[currIndex][1] = 0
        edgeMovies[currIndex][2] = 0 # FIXME 目前先直接把pass的fitness改为0，若要输出方案，则以后再深究
        if(initialSize < leftRoom):
            break
    

    print("新方案剩余空间：" + str(np.sum(edgeMovies[:,1])))
    print("新方案替换后流行度：" + str(np.sum(edgeMovies[:,2])))
    
    return np.sum(edgeMovies[:,2])



def main():

    edgeMovies = generateData(20000)

    # 随机生成视频请求，目前还是限制一下大小，从简
    videoRequest = random.randint(5, VIDEOMAXSIZE)
    leftRoom = np.sum(edgeMovies[:,1]) - videoRequest

    populationsize = 50
    crosstime = 100
    generationtime = 10000
    mutatetime = 100

    # 遗传算法
    GA(edgeMovies, leftRoom, populationsize, crosstime, mutatetime, generationtime)
    dropMinFitness(edgeMovies, leftRoom)





main()
