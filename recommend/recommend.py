import csv
import os
import heapq
import json
import pandas as pd
import numpy as np


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



def readCSV(filePath):
    '''
    func:读取csv的内容，文件头也包含在内
    '''

    try:
        file=open(filePath,'r',encoding="gbk")# 读取以utf-8
        context = file.read() # 读取成str
        list_result=context.split("\n")#  以回车符\n分割成单独的行

        #每一行的各个元素是以【,】分割的，因此可以
        length=len(list_result)

        for i in range(length):
            list_result[i]=list_result[i].split(",")
        return list_result

    except Exception:
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    
    finally:
        file.close();# 操作完成一定要关闭



def toCsv(filename, list1):
    '''
    func:把listMovie数据写入 csv（是list类型的哦，当心别传出array
    '''
    name = ['clickNum', 'avgScore', 'commNum', 'conpreScore']
    test = pd.DataFrame(columns = name, data = list1)
    test.to_csv(dataPath+filename+'.csv',encoding = 'gbk')



def getClicksandAvgScores():
    '''
    func:获取点击量和总评分，结果计入listMovie数组
    '''
    filename = dataPath + "ratings.csv"
    list1 = readCSV(filename)

    for i in range(1,len(list1)-1): #userId,movieId,rating,timestamp 第一行是header，最后一行是空的
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

    for i in range(1,len(list1)-1): #userId,movieId,tag,timestamp
        movieNum = int(list1[i][1])

        listMovie[movieNum][2] = listMovie[movieNum][2] + 1

    print("获取评论数……done")



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
    return: list
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

    print("hot movie index list:   " + str(hotMovieList))

    return hotMovieList # 是个list


def userMartrix(targetUser):

    # 先读文件咯
    filename = dataPath + 'ratings.csv'
    list1 = readCSV(filename) # userId,movieId,rating,timestamp

    # 针对每个用户构造一个矩阵把？

    return 



def main():
    ahpWei = ahp()
    getHotMovies(ahpWei)



main()


# TODO还不就是要接着把用户矩阵搞出来咯！

'''
读取当前目录/父目录的办法，留着备用
    print (os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
    print (os.path.abspath(os.path.dirname(os.getcwd())))
    print (os.path.abspath(os.path.join(os.getcwd(), "..")))
'''
