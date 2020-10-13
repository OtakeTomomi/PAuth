"""
その5 : combine_FCfiles.csvにflag情報を付与するのが目的
"""

import pandas as pd
from tqdm import tqdm

# warning ignore code
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
df_read = pd.read_csv("combine_FC_docfiles2.csv", sep=",")
df_ori = df_read.drop({'Unnamed: 0'}, axis=1)
df = df_ori.copy()
print('欠損値削除前→行数：{0} 列数：{1}'.format(df.shape[0], df.shape[1]))

multi_flag = []
count = 0
for data in tqdm(range(len(df))):
    for i in range(1, 5, 1):
        for j in range(1, 5, 1):
            if df.iloc[data]['flag'] == i:
                if df.iloc[data]['flag2'] == j:
                    new_flag = str(i) + str(j)
                    multi_flag.append(int(new_flag))
                else:
                    new_flag = 100
            else:
                new_flag = 100
    count += 1

print(count)
multi_flag = pd.DataFrame(multi_flag, columns=['multi_flag'])
df = pd.concat([df, multi_flag], axis=1, join='outer')

# ここでcombine_FCfiles.csvの欠損値を削除する
df_last = df.dropna(axis=0, how='any')

print(df.info())
print('欠損値削除後→行数：{0} 列数：{1}'.format(df_last.shape[0], df_last.shape[1]))

# 書き出す時にindex = Falseで消したほうがいい
df_last.to_csv('expdata_doc.csv')
