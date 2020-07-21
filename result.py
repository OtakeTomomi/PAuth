import pandas as pd
import numpy as np

# Columnのリスト
val_columns = ['user', 'flag', 'performance', 'model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']
test_columns = ['user', 'flag', 'performance', 'model', 'AUC', 'Accuracy', 'BER', 'F1', 'FAR', 'FRR', 'Precision', 'Recall']

model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']

# 結果のデータを読み込み
# 交差検証データ
val_df = pd.read_csv("result/result_val.csv", sep=",", header=None)
val_df.columns = val_columns
# テストデータ
test_df = pd.read_csv("result/result_test.csv", sep=",", header=None)
test_df.columns = test_columns

# multi_indexの設定
val_m_df = val_df.set_index(['user', 'flag', 'performance', 'model'])
test_m_df = val_df.set_index(['user', 'flag', 'performance', 'model'])

# 各flagにおいてデータを有するユーザの数
def count_table(df):
    ct_list = sorted(list(set((df.index.get_level_values('flag')))))
    ct = {}
    for i in ct_list:
        ct_n = df['Accuracy'].xs([i, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model']).count()
        ct[i] = ct_n
    return ct, ct_list
ct, ct_list = count_table(val_m_df)
# print(ct)

# 書き出し用の処理
'''
どう分けるかが問題
'''
def write_data(df,model_index,val_columns, flag_n, text):
    flag = pd.Series([flag_n] * 4, name='flag')
    model = pd.Series(model_index, name='model')
    df = pd.DataFrame(df, index=model_index, columns=val_columns)
    print(df)
    df = df.reset_index()
    df = df.drop('index', 1)
    re = pd.concat([flag, model, df], axis=1)
    re.to_csv(f'result/result_{text}.csv', mode='a', header=False, index=False)
    return df
# 各flag × 各modelごとの結果を算出
def calcuration(ct, ct_list,df, val_columns, model_index):
    for k in ct_list:
        a, b, c, d = np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((4, 8)), np.zeros((4, 8))
        for i, model in enumerate(model_index):
            for j, column in enumerate(val_columns[4:]):
                data = df[column].xs([k, 'val', model], level=['flag', 'performance', 'model'])
                a[i][j] = data.mean()
                b[i][j] = data.max()
                c[i][j] = data.min()
                d[i][j] = data.std()

        memori = ['0', 'a', 'b', 'c', 'd']
        # 該当ユーザの抽出
        n = df.xs([k, 'val', 'LocalOutlierFactor'], level=['flag', 'performance', 'model'])
        menber = list(n.index.get_level_values('user'))
        print(f'\n===================================================================================================='
              f'\nflag : {memori[k//10]}+{memori[k%10]}, user数 : {ct[k]}人, 該当ユーザ : {menber}'
              f'\n====================================================================================================')
        print(f'\n平均')
        write_data(a, model_index, val_columns[4:], k, 'mean')
        print(f'\n最大')
        write_data(b, model_index, val_columns[4:], k, 'max')
        print(f'\n最小')
        write_data(c, model_index, val_columns[4:], k, 'min')
        print(f'\n標準偏差')
        write_data(d, model_index, val_columns[4:], k, 'std')

calcuration(ct,ct_list,val_m_df,val_columns,model_index)