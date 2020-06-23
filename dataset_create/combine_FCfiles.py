'''
STEP4 : FC_doc内の全てのcsvファイルを一つに結合する
'''

import numpy as np
import pandas as pd
import os
import glob
import csv

# パスを取得
DATA_PATH = "./FC_doc/"
All_Files = glob.glob('{}*.csv'.format(DATA_PATH))

# フォルダ中の全csvをマージ
list = []
for file in All_Files:
    list.append(pd.read_csv(file))
df = pd.concat(list, sort=False)
del df['Unnamed: 0']

# csv出力
df.to_csv('combine_FC_docfiles.csv')
