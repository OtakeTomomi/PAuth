"""
その4 : FC_doc内の全てのcsvファイルを一つに結合する

"""
import pandas as pd
import glob

# パスを取得
DATA_PATH = "./FC_doc2/"
All_Files = glob.glob('{}*.csv'.format(DATA_PATH))

# フォルダ中の全csvをマージ
f_list = []
for file in All_Files:
    f_list.append(pd.read_csv(file))
df = pd.concat(f_list, sort=False)
del df['Unnamed: 0']

# csv出力
df.to_csv('combine_FC_docfiles2.csv')
