import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
#raw labels
"""这里的data_train.csv文件是抽取了原来训练文件中的X1 Y1 X3 Y3以及filename和type之后的文件，
需要剔除两个label有问题的文件，当然也可以继续剔除其他觉得有问题的label"""
raw_labels = pd.read_csv("data_train.csv",header=None).values
print(raw_labels.shape)#(20238, 6)
filename = []
label = []
for value in raw_labels:
    filename.append(value[0])
    label.append(value[5])

# import collections
# print(collections.Counter(label))
#Counter({'17': 1022, '16': 1019, '18': 1017, '5': 1017, '11': 1014, '10': 1013, '14': 1013, '9': 1013, '20': 1013, '1': 1012, '13': 1011, '6': 1011, '19': 1011, '4': 1010, '15': 1009, '3': 1009, '7': 1008, '2': 1008, '8': 1007, '12': 1002})

X_train, X_val, y_train, y_val = train_test_split(filename, label, test_size=0.2, random_state=42, shuffle=True, stratify=label) #16191
# print((X_train))'f6a2a4aea44a48a09d253538a310af71.jpg'

with open("train_label.csv","w") as f, open("val_label.csv", "w") as g:
    for value in raw_labels:
        filename = os.getcwd() + "/train/" + value[0].split('/')[-1]
        label = str(value[1]) + ',' + str(value[2]) + ',' + str(value[3]) + ',' + str(value[4])
        type = str(value[5])

        if value[0] in X_train:
            dstname1 = os.getcwd() + "/train_pic/" + value[0].split('/')[-1]
            shutil.copy(filename,dstname1)
            f.write(dstname1 + "," + label + ',' + type + "\n")

        else:

            dstname2 = os.getcwd() + "/val_pic/" + value[0].split('/')[-1]
            shutil.copy(filename, dstname2)
            g.write(dstname2 + "," + label + ',' + type + "\n")


