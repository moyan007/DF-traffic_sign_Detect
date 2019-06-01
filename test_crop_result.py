#这个脚本是为了把crop之后网络的得到的结果转换到原图上，可以提交

import pandas as pd
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import cv2

result = pd.read_csv("../result_crop.csv", index_col='filename')               #检测结果

num = 0
with open("detla_test.txt", "r") as f:
    for line in f.readlines():
        pic_name = line.split(' ')[0]#去掉列表中每一个元素的换行符
        detla_x = int(line.split(' ')[1])
        detla_y = int(line.split(' ')[2])

        r = result.loc[pic_name]
        r.X1 += detla_x
        r.X2 += detla_x
        r.X3 += detla_x
        r.X4 += detla_x

        r.Y1 += detla_y
        r.Y2 += detla_y
        r.Y3 += detla_y
        r.Y4 += detla_y

        print("这是第", num, "个, 这是类：", r.type)
        num += 1

    result.to_csv("final_submission.csv")



