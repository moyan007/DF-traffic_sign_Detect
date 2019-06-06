"""当前脚本用于裁剪训练图像
以目标中心为原点 上下左右各扩展200pixel进行切割
"""
import pandas as pd
import shutil
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import cv2

"""这里的train_only.csv文件是抽取了原训练文件train_label_fix.csv中的X1 Y1 X3 Y3以及filename和type之后的文件，剔除清洗掉的明显错误标注的两张图片，
如果发现更多错误标注，剔除了更好"""
df = pd.read_csv("./data/train_only.csv",header=None)
t = open('./data/detla.txt', 'w')

file = open('./data/data_train.csv', "w")
# print(df.shape) #(20239, 6)
for i in tqdm(range(df.shape[0])):
    pic_name = df[0][i]
    # print(pic_name)
    re_pic_name = str('%06d' % i) + '.jpg'            #新的名字补零

    # img = Image.open(pic_name)                        #读取原图
    # img_np = np.array(img)
    # (height, width, deepth) = img_np.shape            #原图的信息
    height,width = 1800,3200
    x_center = int((df[1][i] + df[3][i])/2)           #标注框的中心点
    y_center = int((df[2][i] + df[4][i])/2)

    crop_x_left = x_center - 200                      #四个剪切的边界
    crop_x_right = x_center + 200
    crop_y_up = y_center - 200
    crop_y_down = y_center + 200

    if crop_x_left < 0:                             #防止越界
        crop_x_left = 0
    if crop_y_up < 0:
        crop_y_up = 0
    if crop_x_right > width:
        crop_x_right = width
    if crop_y_down > height:
        crop_y_down = height

    detla_x = crop_x_left                         #记录x, y的偏移量，为了最后的坐标输出
    detla_y = crop_y_up

    t.write(pic_name + ' ' + str(detla_x) + ' ' +  str(detla_y) + '\n')           #记录偏移量

    # new_img = img.crop((crop_x_left,crop_y_up,crop_x_right ,crop_y_down))
    # # new_img.show()
    # new_img.save('./data/train/' + re_pic_name)              #保存剪裁的图片

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

t.close()


