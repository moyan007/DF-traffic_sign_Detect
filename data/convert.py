"""这个脚本用于提取原图上的训练坐标x1 y1 x2 y2形式
train_label_0505.csv  这个文件是提出了两张错误标注后的train_label_fix.csv文件
生成 train_only.csv 和 test_only.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#raw labels
raw_labels = pd.read_csv("train_label_0505.csv",header=None)[1:].values

with open("train_only.csv","w") as f:
    for value in raw_labels:
        filename = os.getcwd() + "/Train_fix/"  +value[0]
        label = str(value[1])+','+str(value[2])+','+str(value[5])+','+str(value[6])
        type = value[9]#-1

        f.write(filename+","+label + ','+type+"\n")

#raw labels
raw_labels = pd.read_csv("submit_sample_fix.csv",header=None)[1:].values

with open("test_only.csv","w") as f:
    for value in raw_labels:
        filename = os.getcwd() + "/Test_fix/"  +value[0]

        f.write(filename+'\n')
