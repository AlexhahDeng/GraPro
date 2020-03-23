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



def generateData():
    '''
    func：在万般无奈下只能自己编造数据了
    return：array col1：点击数 col2：平均评分 col3：评论数 col4：大小
    '''
    # 直接把电影列表作为全局变量把！
    # col1：电影点击数 col2：电影平均评分 col3：电影评论数 col4：综合评分
    listMovie = [[0.0 for col in range(4)] for row in range(MAXNUM)] 
    listMovie = np.asarray(listMovie)

    listMovie = getClicksandAvgScores(listMovie)
    listMovie = getCommentsNum(listMovie)

    # 搞一些随机数据
    # 
    edgeMovies = []
    edgeSize = EDGECAPACITY
    for i in range(len(listMovie)):
        if(listMovie[i,0] == 0):
            continue
        else:
            # 随机生成视频大小
            videoSize = random.randint(5,VIDEOMAXSIZE)
            while(edgeSize - videoSize < 0):
                videoSize = random.randint(5,VIDEOMAXSIZE)

            listMovie[i,3] = videoSize
            edgeMovies.append(list(listMovie[i,:]))
            edgeSize -= videoSize

            if(edgeSize < 5):
                break

    print("随机数据生成！\n" + str(edgeMovies))





    





def main():
    # 模拟节点情况（后期可以改成随机生成的）
    requestVideo = 5 # 视频请求数
    edgeRam = 30 # 边缘节点内存
    ramLeft = edgeRam - requestVideo # 剩余容量

    generateData()


main()
