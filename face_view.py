# -*- coding: utf-8 -*-
# face_view.py 数据可视化
import cv2
import numpy as np

# 指定存放图片的路径
path = 'face'
# 读取像素数据
data = np.loadtxt('cnn_data.csv')

# 按行取数据
for i in range(data.shape[0]):
    face_array = data[i, :].reshape((48, 48)) # reshape
    cv2.imwrite(path + '\\' + '{}.jpg'.format(i), face_array) # 写图片
