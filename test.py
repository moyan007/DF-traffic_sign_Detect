"""测试生成提交检测结果文件，之后要对先前的采集进行还原以及其他后处理
主要更改测试采用的模型及生成csv文件名即可
"""
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import pandas as pd
# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model

model_path = os.path.join('.','models', 'resnet101_11.h5')  #训练的模型
# load retinanet model
model = models.load_model(model_path, backbone_name='resnet101')

# if the model is not converted to an inference model, use the line below
#model = models.convert_model(model)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: '1',1: '2',2: '3',3: '4',4: '5',5: '6',6: '7',7: '8',8: '9',9: '10',10: '11',11: '12',12: '13',13: '14',14: '15',15: '16',16: '17',17: '18',18: '19',19: '20'}
# load image
filename = './output/result_upload_42_48_007.csv'
data = pd.read_csv(filename,header=None,index_col=None)[1:].values
# print(data.shape)
with open("./output/upload/test_upload_11.csv","w") as f:
    f.write('filename' + "," + 'X1'+","+'Y1'+","+'X2'+","+'Y2'+","+'X3'+","+'Y3'+","+'X4'+","+'Y4'+","+"type" + "\n")
    i = 0
    for value in data:
        i += 1
        print(i)
        if(value[9]=='0'):
            print("pass this pic")
            continue
        image = read_image_bgr('./data/test/'+value[0])
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        file = value[0]

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        box, score, label = boxes[0][0],scores[0][0],labels[0][0]

        b = box
        if(score == -1):
            b = [0,0,0,0]
            label = '0'
        else:
            label = labels_to_names[label]
        bbox = str(b[0])+","+str(b[1])+","+str(b[2])+","+str(b[1])+","+\
               str(b[2])+","+str(b[3])+","+str(b[0])+","+str(b[3])
        print(bbox,score,label)
        f.write(file+","+bbox +","+ label+"\n")


print("write done")
