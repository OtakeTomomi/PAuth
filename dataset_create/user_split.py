'''
STEP 1 : data splits by user

featMat.csv is the feature data extracted from the original data.
add it is the data extracted by the creator.
This program splits featMat.csv by user.

'''

# import libraries
import pandas as pd
import numpy as np
import os

# Reading featMat.csv
fM = pd.read_csv('../02_features/featMat.csv', delimiter = ",", header = None)

# fMというフォルダの作成
os.makedirs('fM', exist_ok = True)

# 列0の値ごとに抽出してfMフォルダ内に保存
count = 0
for x in range(1, 42):
    fM2 = fM[fM[0] == x]
    fM2.to_csv("fM/fM{}.csv".format(x), header = None)
    count += 1
print('ループ回数：{}'.format(count))