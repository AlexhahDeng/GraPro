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
from func import *
import datetime
np.set_printoptions(suppress=True)

# 数据路径
dataPath=os.path.abspath(os.path.dirname(os.getcwd())) + "\data\\" 

# 电影数(只有在换数据集的时候才需要改)
MAXNUM = 1000000

# KEYPOINT 热门电影数
hotMovieNum = 200

# 直接把电影列表作为全局变量把！col1：电影点击数 col2：电影平均评分 col3：电影评论数 col4：综合评分
listMovie = [[0.0 for col in range(4)] for row in range(MAXNUM)] 
listMovie = np.asarray(listMovie)

# 用户集合
listUser = [[0.0 for col in range(4)] for row in range(MAXNUM)]
listUser = np.asarray(listUser)

# ahp一致性检验参数，阶为3的情况，如果后期要加还要根据情况变化
RI = 0.58

# 最小的负数
minNum = -sys.maxsize

# KEYPOINT 推荐电影数，可以看成边缘节点容量限制
EDGEMAXMOVIES =500

def getUserMovies():
    '''
    func：搞到每个用户的观影信息，list的index是userid，内容是一个电影id+评分+平均评分的二维数组
    return：list--col1:id, col2:movielist(str), col3: avg scores
    '''
    # 读文件啦
    filename = dataPath + 'ratings.csv'
    listRating = readCSV(filename)

    # 整理数据啦
    arr = np.asarray(listRating)
    col1 = [int(i[0]) for i in listRating] # 读取第一列
    col1 = np.asarray(col1)
    userList = [ [] for i in range(np.max(col1) + 1)] # 构造用户list

    for i in range(np.max(col1) + 1): 
        movieList = arr[np.where(col1 == i)] # 找到某一id用户的看过的所有电影的集合
        movieList = movieList[:,[1,2]]

        if(len(movieList) == 0): # 该用户没看过电影
            continue
        else:
            totalScores = 0.0
            newMovieList = [] # 为了转换格式，因为原来的movieList里面都是str类型
            for j in range(len(movieList)):
                totalScores = totalScores + float(movieList[j][1])
                newMovieList.append(list(map(eval, movieList[j]))) # str转为int

            avgScores = totalScores / len(movieList) # 该用户打分的平均值
            tmp = [i, newMovieList, avgScores]
            userList[i] = tmp # 还是把用户id和list index对上号

    # print(userList[1])
    # print("获取每个用户的观影记录以及评分……done\n")

    return userList 


def getHotMovies(ahp, mvlist, num):
    '''
    func:
        1、根据ahp结果，统计每个电影的综合评分
        2、根据分数高低，获取top-n个电影
    input: ahp是权重；mvlist是待评价的电影list；Num是电影数目
    points：
        1、可以改变top-n的数量来看数据变化
    return: array--col1是movie id,col2是score
    '''

    # 先给我留一个原始的listMovie中的原始平均评分数据 涉及拷贝的问题--用list()函数可解决
    AvgScore = list(mvlist[:, 1])

    # 归一化每一列的数据
    colMax = np.max(mvlist, axis = 0)

    # 获取综合评分
    [rows, cols] = mvlist.shape # 获取电影数组的行列信息

    for i in range(cols): # 外围是列，而不是传统的行，便于一次性处理好所有电影数据
        for j in range(rows):
            if(i == cols-1): # 若是最后一列，则开始计算综合评分
                for k in range(cols - 1):
                    mvlist[j][i] = mvlist[j][k]*ahp[k] + mvlist[j][i]

            else: # 非最后一列，则通过除以列最大值，归一化数据
                mvlist[j][i] = mvlist[j][i] / colMax[i]

    # toCsv('movieList',listMovie.tolist()) 做测试用

    # 获取最热门的n个电影以及评分
    conpreScores = mvlist[:, cols - 1]
    conpreScores = conpreScores.tolist()

    hotMovieIndex = map(conpreScores.index, heapq.nlargest(num, conpreScores)) # 获取综合评分最高的前hotMovieNum个movie id

    hotMovieIndex = list(hotMovieIndex)
    hotMovieList = [[0.0 for col in range(2)] for row in range(hotMovieNum)]

    for i in range(hotMovieNum): # 获取对应movie id 的平均评分
        hotMovieList[i][0] = hotMovieIndex[i] # 序号
        hotMovieList[i][1] = AvgScore[hotMovieIndex[i]] # FIXME 不知道要不要把平均分数化成int，毕竟一般评分基本都是int


    return np.asarray(hotMovieList) # 是个array，col1是movie id,col2是score

def getRecommMovies(targetUser, userMovieList, movienum): # FIXME 这里还要再研究一下正确性
    '''
    func：
        1、获取用户相似度矩阵
        2、然后选择前k个用户补全用户针对电影的评分矩阵
        3、对电影评分矩阵排序获取前n个就是推荐的电影
    points：
        1、考虑是否从推荐电影中去掉热门电影
        2、可以改变推荐的数量
    return：array, movie id
    '''

    MAXK = 50 # KEYPOINT 选择前k个用户，可以做修改

    userSimMartrix = [0.0 for row in range(len(userMovieList) + 1)] # 初始化用户相似度矩阵
    
    for i in range(1, len(userMovieList)): # 遍历所有用户，第一个是空的，index和id对上号了，否则访问空[]会报错哦

        # 获取每个用户基本信息(col1:userid;col2:movie list;col3:avg scores)
        userId = userMovieList[i][0]
        userMovies = np.asarray(userMovieList[i][1]) # movieid + scores
        vuserAvgScores = userMovieList[i][2]
        

        commonMovies = np.intersect1d(targetUser[:, 0], userMovies[:, 0]) # 找到目标用户和用户一起看过的电影
        totalMovies = np.union1d(targetUser[:, 0], userMovies[:, 0]) # 二者看过的所有电影之和（无重复）
        tuserAvgScores = np.sum(targetUser[:, 1]) / len(targetUser) # 目标用户的平均评分

        sim = 0.0 # 用户相似度

        # print(commonMovies)
        # print( "target user avg score   " + str(tuserAvgScores))
        # print( "total movies " + str(len(totalMovies)))

        for j in range(len(commonMovies)):
            # 获取电影评分
            movieId = int(commonMovies[j])

            uArr = targetUser[np.where(targetUser[:,0] == movieId)] 
            vArr = userMovies[np.where(userMovies[:, 0] == movieId)]

            u = uArr[0][1] # target user score
            v = vArr[0][1] # user score
            
            # FIXME 有个问题是，我不知道这里到底算的对不对……

            # 用户评分相似度
            sim1 = 2 * (1 - 1 / (1 + math.exp(- abs(u - v))))

            # 用户偏好相似度
            sim2 = 2 * (1 - 1 / (1 + math.exp(- abs((u - tuserAvgScores) - (v - vuserAvgScores)))))

            # 置信度
            sim3 = len(commonMovies) / len(totalMovies)
            # print(str(s1) + "/" + str(s2) + "/" + str(s3))

            sim = sim + sim1 * sim2 * sim3

        if (len(commonMovies) == 0): # 没有看过一样的电影
            totalSim = 0
        else:
            totalSim = sim/len(commonMovies) # 这就是用户相似度了！！！

        userSimMartrix[userId] = totalSim    

    # print(userSimMartrix)

    # 获取前k个最相似用户
    preKUserIndex = map(userSimMartrix.index, heapq.nlargest(MAXK, userSimMartrix))
    preKUserIndex = list(preKUserIndex) # 这里得到的是index
    allUserSim = 0 # 最相似用户的相似度之和

    movieScoreList = [ [] for i in range(MAXNUM)] # 初始化一个list的list，用来放每个电影的……不知道咋描述

    for i in range(len(preKUserIndex)):

        userId = preKUserIndex[i]
        userSim = userSimMartrix[userId] # 用户相似度
        allUserSim = allUserSim + userSim

        userMovies = userMovieList[userId][1] # 获取该用户的观影评分记录
        userAvgScore = userMovieList[userId][2] # 用户平均评分

        for j in range(len(userMovies)): # 遍历用户观影记录
            movieId = userMovies[j][0]
            movieScore = userMovies[j][1]
            movieScoreList[movieId].append((movieScore - userAvgScore) * userSim)

    # 统计综合预测评分了！！
    for i in range(len(movieScoreList)):
        if(movieScoreList[i] == []):
            movieScoreList[i] = minNum
        else:
            tmp = np.asarray(movieScoreList[i])
            movieScoreList[i] = np.sum(tmp)/allUserSim
            # print(movieScoreList[i])

    # FIXME 有个问题，这里统计的最高分的电影是不是得pass掉热门电影集的内容。
    # 获取预测评分最高的电影
    hotMoviesPre = map(movieScoreList.index, heapq.nlargest(movienum, movieScoreList))

    return np.asarray(list(hotMoviesPre))

def analyze(recoMovies, hotMovies, userMovies):
    '''
    return: 推荐电影命中率，最流行命中率
    '''
    # print("推荐电影命中率……\n")
    recoHitRate = len(np.intersect1d(recoMovies, userMovies)) / len(userMovies)
    # print(str(recoHitRate) + "\n")

    # print("最流行存储命中率……\n")
    hotHitRate = len(np.intersect1d(hotMovies, userMovies)) / len(userMovies)
    # print(str(hotHitRate) + "\n")

    # print(str(recoMovies)+"\n"+str(hotMovies)+"\n"+str(userMovies))

    return recoHitRate, hotHitRate


def hitRate(movieList, allMovies):

    return len(np.intersect1d(movieList, allMovies)) / len(allMovies)


def getuserSeperate(edgeUserId, userMovieList):
    '''
    func：尝试新方法来获取推荐内容，每个用户单独推荐，获取推荐矩阵
    return：movie id(array)
    '''

    # 获取边缘用户记录
    movieList = [[0 for i in range(2)] for j in range(MAXNUM)]

    # 针对每个单独的用户获得推荐电影
    for i in range(len(edgeUserId)):

        target_user_id = int(edgeUserId[i])
        target_user_movie = userMovieList[target_user_id][1]

        recommMovieList = getRecommMovies(np.asarray(target_user_movie), userMovieList, EDGEMAXMOVIES)

        for j in range(len(recommMovieList)):
            movieId = int(recommMovieList[j])
            # if(movieId == 0): # BUG 可能是有个随机bug…whatever管他呢
            #     print("ono")
            #     break
            movieList[movieId][0] = movieId
            movieList[movieId][1] += 1
        
        
    movieList = np.asarray(movieList)
    movieIndex = movieList[movieList[:,1].argsort()]

    # 选择缓存的电影数量
    movieIndex = movieIndex[-EDGEMAXMOVIES:]
    movieIndex = movieIndex[:,0] #只要第一列

    # toCSV("hotmovies", ["clicknum"], movieList, True)
    return movieIndex
    
def getRecoMovies(edgeUserMovieList, userMovieList):
    '''
    func：
        1、获取把边缘节点所有用户看作一个整体来进行推荐的结果
        2、获取边缘区域最流行缓存的结果

    return: reco movies(movie id),popular movies(movie id) -- array
    '''

    ahpwei = ahp()

    # 获取热门电影集(这里的热门电影集合是局部的)
    targetUser = getHotMovies(ahpwei, edgeUserMovieList, EDGEMAXMOVIES)

    # 获取局部热门电影用作最流行缓存
    mostPopMovies = targetUser[:,0]

    # 获取推荐电影集
    recommMovieList = getRecommMovies(targetUser, userMovieList, EDGEMAXMOVIES)

    return recommMovieList, mostPopMovies

def getWWPopMovies():

    ahpwei = ahp()

    getClicksandAvgScores(listMovie)
    getCommentsNum(listMovie)

    movieList = getHotMovies(ahpwei, listMovie, EDGEMAXMOVIES)

    return movieList[:,0]

def main():

    # 边缘节点用户数量
    edgeUserNum = 50

    # 所有用户观影记录
    userMovieList = getUserMovies()

    # 随机生成边缘用户
    edgeUserId = generateEdgeUser(edgeUserNum)

    # 获取边缘节点用户观影记录
    edgeUserMovieList, testMovies = edgeUser(edgeUserId)    

    # 获取针对独立用户的推荐结果
    userSeperate = getuserSeperate(edgeUserId, userMovieList)
    print("用户分开推荐结果     " + str(hitRate(userSeperate, testMovies)))

    # 将所有边缘用户看成模拟为一个用户进行推荐
    userAsOne, hotMovies = getRecoMovies(edgeUserMovieList, userMovieList)
    print("用户看成整体推荐结果     " + str(hitRate(userAsOne, testMovies)))
    print("区域最流行缓存       " + str(hitRate(hotMovies, testMovies)))

    # 获取全局最流行缓存
    wwpopmovies = getWWPopMovies()
    print("全局最流行缓存        " + str(hitRate(wwpopmovies, testMovies)))

    analyze(userSeperate, userAsOne, testMovies)

    return


def testUserNum():
    '''
    func：判断区域用户数量多少的影响
    '''
    test_result = []
    # 所有用户观影记录
    userMovieList = getUserMovies()

    for i in range(5,100):
        curr = []

        # 获取模拟数
        edgeUserNum = i * 5
        edgeUserId = generateEdgeUser(edgeUserNum)
        edgeUserMovieList, testMovies = edgeUser(edgeUserId)

        if(len(testMovies) == 0):
            i -= 1
            continue # 莫得测试数据
        print("usernum" + str(edgeUserNum))

        # 获取边缘节点用户观影记录
        edgeUserMovieList, testMovies = edgeUser(edgeUserId)    

        # 获取针对独立用户的推荐结果
        userSeperate = getuserSeperate(edgeUserId, userMovieList)

        # 将所有边缘用户看成模拟为一个用户进行推荐
        userAsOne, hotMovies = getRecoMovies(edgeUserMovieList, userMovieList)

        # 获取全局最流行缓存
        wwpopmovies = getWWPopMovies()

        curr.append(edgeUserNum)
        curr.append(hitRate(userSeperate, testMovies))
        curr.append(hitRate(userAsOne, testMovies))
        curr.append(hitRate(hotMovies, testMovies))
        curr.append(hitRate(wwpopmovies, testMovies))

        test_result.append(curr)
    
    toCSV("usernum2",["num","seperate","asOne","edgehotMovies", "worldwideHotMovies"],test_result)

testUserNum()