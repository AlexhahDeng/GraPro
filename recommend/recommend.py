import csv
import os
import numpy as np


# 数据路径
dataPath=os.path.abspath(os.path.dirname(os.getcwd())) + "\data\\" 

# 电影数
MAXNUM = 1000000

# 直接把电影列表作为全局变量把！col1：电影点击数 col2：电影平均评分 col3：电影评论数
listMovie = [[0 for col in range(3)] for row in range(MAXNUM)] 

# ahp一致性检验参数，阶为3的情况
RI = 0.58

# 读取csv的内容
def readCSV(filePath):
    # 文件头也包含在内
    try:
        file=open(filePath,'r',encoding="gbk")# 读取以utf-8
        context = file.read() # 读取成str
        list_result=context.split("\n")#  以回车符\n分割成单独的行

        #每一行的各个元素是以【,】分割的，因此可以
        length=len(list_result)

        for i in range(length):
            list_result[i]=list_result[i].split(",")
        return list_result,

    except Exception:
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    
    finally:
        file.close();# 操作完成一定要关闭


# 获取点击量和总评分
def getClicksandAvgScores():

    filename = dataPath + "ratings.csv"
    list1 = readCSV(filename)

    for i in range(1,len(list1)-1): #userId,movieId,rating,timestamp 第一行是header，最后一行是空的
        movieNum = int(list1[i][1]) # movieid
        
        listMovie[movieNum][0] = listMovie[movieNum][0] + 1 # count clicks
        listMovie[movieNum][1] = float(list1[i][2]) + listMovie[movieNum][1] # count scores

    for i in range(len(listMovie)):
        if(listMovie[i][0] != 0):
            listMovie[i][1] = listMovie[i][1] / listMovie[i][0]


# 获取评论数
def getCommentsNum():

    filename = dataPath + "tags.csv"
    list1 = readCSV(filename)

    for i in range(1,len(list1)-1): #userId,movieId,tag,timestamp
        movieNum = int(list1[i][1])

        listMovie[movieNum][2] = listMovie[movieNum][2] + 1


def ahp():
    '''
        看过的一定会评分，但是不一定会打tag
        把ratings中电影出现次数看成 “点击量” 评分取 “平均评分”
        把tags中某个电影被打tag的次数看成 “评论数”
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







ahp()








'''
读取当前目录/父目录的办法，留着备用
    print (os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
    print (os.path.abspath(os.path.dirname(os.getcwd())))
    print (os.path.abspath(os.path.join(os.getcwd(), "..")))
'''
