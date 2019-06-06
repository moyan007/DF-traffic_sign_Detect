#训练集图像随机裁剪脚本，保证目标在裁剪后的图像上即可，从而进行数据增强，可增强多倍数据
import pandas as pd
import shutil
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import cv2
import random

df = pd.read_csv("./data/train_only.csv",header=None)
# t = open('./data/detla.txt', 'w')

file = open('./data/data_train.csv', "w")
print(df.shape) #(20239, 6)

for j in range(5):                                          #扩充5倍数据量
    for i in tqdm(range(df.shape[0])):
        pic_name = df[0][i]
        # print(pic_name)
        re_pic_name = str(i) + '_' + str(j) + '.jpg'     #新的名字补零

        img = Image.open(pic_name)                        #读取原图
        img_np = np.array(img)
        (height, width, deepth) = img_np.shape            #原图的信息
        # height,width = 1800,3200
        x_center = int((df[1][i] + df[3][i])/2)            #标注框的中心点
        y_center = int((df[2][i] + df[4][i])/2)

        x_center += random.randint(-80, 80)                #中心点加随机数
        y_center += random.randint(-80, 80)
        if x_center < 0:                                   #防止越界
            x_center = 0
        if x_center > width:
            x_center = width
        if y_center < 0:
            y_center = 0
        if y_center > height:
            y_center = height

        crop_x_left = x_center - 200                      #四个剪切的边界
        crop_x_right = x_center + 200
        crop_y_up = y_center - 200
        crop_y_down = y_center + 200

        if crop_x_left < 0:                              #防止越界
            crop_x_left = 0
        if crop_y_up < 0:
            crop_y_up = 0
        if crop_x_right > width:
            crop_x_right = width
        if crop_y_down > height:
            crop_y_down = height

        detla_x = crop_x_left                         #记录x, y的偏移量，为了最后的坐标输出
        detla_y = crop_y_up

        # t.write(pic_name + ' ' + str(detla_x) + ' ' +  str(detla_y) + '\n')           #记录偏移量

        new_img = img.crop((crop_x_left,crop_y_up,crop_x_right ,crop_y_down))  #截图
        # new_img.show()
        new_img.save('./data/train/' + re_pic_name)              #保存剪裁的图片

        xmin = df[1][i] - detla_x
        ymin = df[2][i] - detla_y
        xmax = df[3][i] - detla_x
        ymax = df[4][i] - detla_y
        file_name = os.getcwd() + '/data/train/' + re_pic_name
        # print(type(xmin))
        file.write(file_name+','+str(xmin)+','+str(ymin)+','+str(xmax)+','+
                   str(ymax)+','+str(df[5][i])+'\n')
    # img_show = cv2.imread('./data/train/' + re_pic_name)
    # cv2.rectangle(img_show, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    # plt.imshow(img_show)
    # plt.show()
