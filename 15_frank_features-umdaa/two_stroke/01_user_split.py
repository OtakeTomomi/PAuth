"""
その1 : userごとにデータを分割
featMat.csv is the feature data extracted from the original data.
add it is the data extracted by the creator.
This program splits featMat.csv by user.
"""

# import libraries
import pandas as pd
import os
import time

print('STEP 1: start')
# 開始
start_time = time.perf_counter()

# Reading featMat.csv
fM = pd.read_csv('../one_stroke/frank_features_calc.csv', delimiter=",", header=None)

# fMというフォルダの作成
# fu = feank-umdaa
os.makedirs('featMat_fu', exist_ok=True)

# 列0の値ごとに抽出してfMフォルダ内に保存
count = 0
for x in range(1, 42):
    # userのColumnは０
    fM2 = fM[fM[0] == x]
    fM2.to_csv("featMat_fu/featMat_fu{}.csv".format(x), header=None)
    count += 1
print('ループ回数：{}'.format(count))

# 終了
end_time = time.perf_counter()

# 経過時間を出力(秒)
elapsed_time = end_time - start_time
print(elapsed_time)

print('STEP 1: OK')