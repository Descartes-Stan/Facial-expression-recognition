#将label和像素数据分离
import pandas as pd

path = 'cnn_train.csv'# 原数据路径
# 读取数据
df = pd.read_csv(path)
# 提取label数据
df_y = df[['label']]
# 提取feature（即像素）数据
df_x = df[['feature']]
# 将label写入label.csv
df_y.to_csv('cnn_label.csv', index=False, header=False)
# 将feature数据写入data.csv
df_x.to_csv('cnn_data.csv', index=False, header=False)
