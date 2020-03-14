import csv
import os
import sys
import heapq
import json
import math
import pandas as pd
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

# TODO 基本上搞完了，再找个时间整理下逻辑，然后规整一下代码，最好是能把一些参数，名字啥的都给成立好，省的以后要改又一汤子糊！

# 数据路径
dataPath=os.path.abspath(os.path.dirname(os.getcwd())) + "\data\\" 

# 电影数
MAXNUM = 1000000

# 热门电影数
hotMovieNum = 100

# 直接把电影列表作为全局变量把！col1：电影点击数 col2：电影平均评分 col3：电影评论数 col4：综合评分
listMovie = [[0.0 for col in range(4)] for row in range(MAXNUM)] 
listMovie = np.asarray(listMovie)

# 用户集合
listUser = [[0.0 for col in range(4)] for row in range(MAXNUM)]
listUser = np.asarray(listUser)

# ahp一致性检验参数，阶为3的情况
RI = 0.58

# 最小的负数
minNum = -sys.maxsize



def readCSV(filePath):
    '''
    func:读取csv的内容
    '''

    try:
        file=open(filePath,'r',encoding="gbk")# 读取以utf-8
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



def toCsv(filename, list1):
    '''
    func:把listMovie数据写入 csv（是list类型的哦，当心别传入array
    '''
    name = ['clickNum', 'avgScore', 'commNum', 'conpreScore']
    test = pd.DataFrame(columns = name, data = list1)
    test.to_csv(dataPath+filename+'.csv',encoding = 'gbk')



def getClicksandAvgScores():
    '''
    func:获取点击量和平均评分，结果计入listMovie数组
    '''
    filename = dataPath + "ratings.csv"
    list1 = readCSV(filename)

    for i in range(len(list1)): #userId,movieId,rating,timestamp 第一行是header，最后一行是空的
        movieNum = int(list1[i][1]) # movieid
        
        listMovie[movieNum][0] = listMovie[movieNum][0] + 1 # count clicks
        listMovie[movieNum][1] = float(list1[i][2]) + listMovie[movieNum][1] # count scores

    for i in range(len(listMovie)):
        if(listMovie[i][0] != 0):
            listMovie[i][1] = listMovie[i][1] / listMovie[i][0]

    print("获取点击量和平均评分……done")



def getCommentsNum():
    '''
    func: 获取评论数，结果直接计入listMovie数组
    '''

    filename = dataPath + "tags.csv"
    list1 = readCSV(filename)

    for i in range(len(list1)): #userId,movieId,tag,timestamp
        movieNum = int(list1[i][1])

        listMovie[movieNum][2] = listMovie[movieNum][2] + 1

    print("获取评论数……done")


def getUserMovies():
    '''
    func：搞到每个用户的观影信息，list的index是userid，内容是一个电影id+评分+平均评分的二维数组
    return：list
    '''
    # 读文件啦
    filename = dataPath + 'ratings.csv'
    listRating = readCSV(filename)

    # 整理数据啦
    arr = np.asarray(listRating)
    col1 = [int(i[0]) for i in listRating] # 读取第一列
    col1 = np.asarray(col1)
    userList = [ [] for i in range(np.max(col1) + 1)]

    print("获取每个用户的观影记录以及评分……done")
    for i in range(np.max(col1) + 1): 
        movieList = arr[np.where(col1 == i)]
        movieList = movieList[:,[1,2]]

        if(len(movieList) == 0):
            continue
        else:
            totalScores = 0.0
            newMovieList = [] # 为了转换格式，因为原来的movieList里面都是str类型
            for j in range(len(movieList)):
                totalScores = totalScores + float(movieList[j][1])
                newMovieList.append(list(map(eval, movieList[j])))

            avgScores = totalScores / len(movieList)
            tmp = [i, newMovieList, avgScores]
            userList[i] = tmp
    # print(userList[1])

    return userList # col1:id, col2:movielist(str), col3: avg scores


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

    if(CR < 0.1):
        print("CR:" + str(CR) +"    通过一致性检验")
    else:
        print("未通过一致性检验，请重新设计对比矩阵")

    return w # 权重array



def getHotMovies(ahp):
    '''
    func:获取电影综合评分后给出热门电影集
    return: array
    '''

    getClicksandAvgScores()
    getCommentsNum()

    # 先给我留一个原始的listMovie中的原始平均评分数据 涉及拷贝的问题--用list()函数可解决
    AvgScore = list(listMovie[:, 1])

    # 归一化每一列的数据
    colMax = np.max(listMovie, axis = 0)
    print("每一列的最大值：" + str(colMax))

    # 获取综合评分
    [rows, cols] = listMovie.shape # 获取电影数组的行列信息

    for i in range(cols): # 外围是列，而不是传统的行，便于一次性处理好所有电影数据
        for j in range(rows):
            if(i == cols-1): # 若是最后一列，则开始计算综合评分
                for k in range(cols - 1):
                    listMovie[j][i] = listMovie[j][k]*ahp[k] + listMovie[j][i]

            else: # 非最后一列，则通过除以列最大值，归一化数据
                listMovie[j][i] = listMovie[j][i] / colMax[i]

    # toCsv('movieList',listMovie.tolist()) 

    # 获取最热门的n个电影以及评分
    conpreScores = listMovie[:, cols - 1]
    conpreScores = conpreScores.tolist()

    hotMovieIndex = map(conpreScores.index, heapq.nlargest(hotMovieNum, conpreScores)) # 获取综合评分最高的前hotMovieNum个movie id

    hotMovieIndex = list(hotMovieIndex)
    hotMovieList = [[0.0 for col in range(2)] for row in range(hotMovieNum)]

    for i in range(hotMovieNum): # 获取对应movie id 的平均评分
        hotMovieList[i][0] = hotMovieIndex[i] # 序号
        hotMovieList[i][1] = AvgScore[hotMovieIndex[i]] # FIXME 不知道要不要把平均分数化成int，毕竟一般评分基本都是int

    # print("hot movie index list:   " + str(hotMovieList))

    return np.asarray(hotMovieList) # 是个array，col1是movie id,col2是score


def userMartrix(targetUser, userMovieList):
    '''
    func：获取用户相似度矩阵，然后选择前k个用户补全用户评分矩阵，最后获得推荐电影
    return：array
    '''

    MAXK = 50 # 选择前k个用户
    MAXMOVIES =100 # 推荐电影数

    userSimMartrix = [0.0 for row in range(len(userMovieList) + 1)] # 初始化用户相似度矩阵
    
    for i in range(1, len(userMovieList)): # 遍历所有用户，第一个是空的，index和id对上号了

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

    # 获取预测评分最高的电影
    hotMoviesPre = map(movieScoreList.index, heapq.nlargest(MAXMOVIES, movieScoreList))

    print(list(hotMoviesPre))

    return list(hotMoviesPre)



def main():
    ahpWei = ahp()
    targetUser = getHotMovies(ahpWei)

    userMovieList = getUserMovies()
    userMartrix(targetUser, userMovieList)


main()


