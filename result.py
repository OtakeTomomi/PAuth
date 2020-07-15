import pandas as pd

# Columnのリスト
val_columns = ['user','flag','performance','model','Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
test_columns = ['user','flag','performance','model', 'AUC', 'Accuracy', 'BER', 'F1', 'FAR', 'FRR', 'Precision', 'Recall']

# 結果のデータを読み込み
val_df = pd.read_csv("result/result_val.csv", sep = ",", header=None)
val_df.columns = val_columns

test_df = pd.read_csv("result/result_test.csv", sep = ",", header=None)
test_df.columns = test_columns

# multi_indexの設定
val_m_df = val_df.set_index(['user', 'flag','performance', 'model'])
test_m_df = val_df.set_index(['user', 'flag','performance', 'model'])

# print(df_m)
# print(df_m.loc[[23],[11,33],['LocalOutlierFactor']])

# 理想は結果を一発表示する
# a = np.zeros((4, 8))