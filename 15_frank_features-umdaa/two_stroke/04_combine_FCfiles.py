"""
その4 : FC_doc内の全てのcsvファイルを一つに結合する

"""
import pandas as pd
import glob
import time

print('STEP 4: start')

# 開始
start_time = time.perf_counter()

# パスを取得
DATA_PATH = "./feature_calc_doc/"
All_Files = glob.glob('{}*.csv'.format(DATA_PATH))

# フォルダ中の全csvをマージ
f_list = []
for file in All_Files:
    f_list.append(pd.read_csv(file))
df = pd.concat(f_list, sort=False)
del df['Unnamed: 0']

# csv出力
df.to_csv('combine_feature_calc_doc_files.csv')


# 終了
end_time = time.perf_counter()

# 経過時間を出力(秒)
elapsed_time = end_time - start_time
print(elapsed_time)

print('STEP 4: OK')
