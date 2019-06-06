"""当前脚本用于裁剪测试图像
以（之前线上检测到框的）目标中心为原点 上下左右各扩展200pixel进行切割，对于不能检测到的图像只能通过再手工标注或者剔除标记为0"""

import pandas as pd
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import cv2

df = pd.read_csv("./output/result_upload_42_48_007.csv")      #初次检测结果
t = open('detla_test.txt', 'w')
# print(df.shape)#(20256, 10)
for i in tqdm(range(df.shape[0])):
    pic_name = df.filename[i]
    if(df.type[i]==0):
        print("pass this pic")
        continue
    # re_pic_name = str('%06d' % i) + '.jpg'                  #新的名字补零

    img = Image.open('./data/Test_fix/' + pic_name)                        #读取原图
    img_np = np.array(img)
    (height, width, deepth) = img_np.shape                     #原图的信息

    x_center = int((df.X2[i] + df.X1[i])/2)                    #标注框的中心点
    y_center = int((df.Y1[i] + df.Y3[i])/2)

    crop_x_left = x_center - 200                               #四个剪切的边界
    crop_x_right = x_center + 200
    crop_y_up = y_center - 200
    crop_y_down = y_center + 200

    if crop_x_left < 0:                                        #防止越界
        crop_x_left = 0
    if crop_y_up < 0:
        crop_y_up = 0
    if crop_x_right > width:
        crop_x_right = width
    if crop_y_down > height:
        crop_y_down = height

    detla_x = crop_x_left                                      #记录x, y的偏移量，为了最后的坐标输出
    detla_y = crop_y_up

    t.write(pic_name + ' ' + str(detla_x) + ' ' +  str(detla_y) + '\n')           #记录偏移量
    new_img = img.crop((crop_x_left,crop_y_up,crop_x_right ,crop_y_down))
    # new_img.show()
    new_img.save('./data/test/' + pic_name)                    #保存剪裁的图片

t.close()
