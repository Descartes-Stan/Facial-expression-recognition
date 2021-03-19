# -*- coding: utf-8 -*-
# 表情图片和类别标注
import os
import pandas as pd


def data_label(path):
    # 读取label文件
    df_label = pd.read_csv('cnn_label.csv', header=None)
    # 查看该文件夹下所有文件
    files_dir = os.listdir(path)
    # 用于存放图片名
    path_list = []
    # 用于存放图片对应的label
    label_list = []
    # 遍历该文件夹下的所有文件
    for file_dir in files_dir:
        # 如果某文件是图片，则将其文件名以及对应的label取出，分别放入path_list和label_list这两个列表中
        if os.path.splitext(file_dir)[1] == ".jpg":
            path_list.append(file_dir)
            index = int(os.path.splitext(file_dir)[0])
            label_list.append(df_label.iat[index, 0])

    # 将两个列表写进dataset.csv文件
    path_s = pd.Series(path_list)
    label_s = pd.Series(label_list)
    df = pd.DataFrame()
    df['path'] = path_s
    df['label'] = label_s
    df.to_csv(path + '\\dataset.csv', index=False, header=False)


def main():
    # 指定文件夹路径
    train_path = 'cnn_train'
    val_path = 'cnn_val'
    data_label(train_path)
    data_label(val_path)


if __name__ == "__main__":
    main()


