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



def main():
    # 模拟节点情况（后期可以改成随机生成的）
    requestVideo = 5 # 视频请求数
    edgeRam = 30 # 边缘节点内存
    ramLeft = edgeRam - requestVideo # 剩余容量