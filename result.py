import pandas as pd

# Columnのリスト
columns = ['user','flag','model','Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']

# 結果のデータを読み込み
r_df = pd.read_csv("result/result.csv", sep = ",", header=None)
r_df.columns = columns

# multi_indexの設定
df_m = r_df.set_index(['user', 'flag', 'model'])

print(df_m)

print(df_m.loc[[23],[11,33],['LocalOutlierFactor']])