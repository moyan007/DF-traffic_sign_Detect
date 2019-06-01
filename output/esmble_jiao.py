#分别尝试求交集与并集来进行融合
import pandas as pd
from tqdm import tqdm

result1 = pd.read_csv("test_upload_42.csv")
result2 = pd.read_csv("test_upload_48.csv")

# 1.交集
for i in tqdm(range(result1.shape[0])):
    new_xmin = max(result1['X1'][i], result2['X1'][i])
    new_xmax = min(result1['X2'][i], result2['X2'][i])
    new_ymin = max(result1['Y1'][i], result2['Y1'][i])
    new_ymax = min(result1['Y4'][i], result2['Y4'][i])

    result1['X1'][i] = new_xmin
    result1['X2'][i] = new_xmax
    result1['X3'][i] = new_xmax
    result1['X4'][i] = new_xmin

    result1['Y1'][i] = new_ymin
    result1['Y2'][i] = new_ymin
    result1['Y3'][i] = new_ymax
    result1['Y4'][i] = new_ymax

result1.to_csv('result_upload_jiao.csv', index=0)