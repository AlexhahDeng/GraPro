import csv
import os

# 数据路径
dataPath=os.path.abspath(os.path.dirname(os.getcwd())) + "\data\\" 

# 电影数
MAXNUM = 1000000

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
        return list_result

    except Exception:
        print("文件读取转换失败，请检查文件路径及文件编码是否正确")
    
    finally:
        file.close();# 操作完成一定要关闭



def getClicksandAvgScores():

    filename = dataPath + "ratings.csv"
    list1 = readCSV(filename)
    listMovie = [[0 for col in range(2)] for row in range(MAXNUM)] # 二维数组，第一列是电影点击数，第二列是电影分数

    for i in range(1,len(list1)): #userId,movieId,rating,timestamp
        movieNum = int(list1[i][1]) # movieid
        
        listMovie[movieNum][0] = listMovie[movieNum][0] + 1 # count clicks
        listMovie[movieNum][1] = float(list1[i][2]) + listMovie[movieNum][1] # count scores

    for i in range(len(listMovie)):
        listMovie[i][1] = listMovie[i][1] / listMovie[i][0]

        print (listMovie[i][0] + "   " + listMovie[i][1])

    return listMovie


def getCommentsNum():

    return 


def ahp():
    '''
        看过的一定会评分，但是不一定会打tag
        把ratings中电影出现次数看成 “点击量” 评分取 “平均评分”
        把tags中某个电影被打tag的次数看成 “评论数”
    '''


getClicksandAvgScores()








'''
读取当前目录/父目录的办法，留着备用
    print (os.path.abspath(os.path.dirname(os.path.dirname('__file__'))))
    print (os.path.abspath(os.path.dirname(os.getcwd())))
    print (os.path.abspath(os.path.join(os.getcwd(), "..")))
'''
