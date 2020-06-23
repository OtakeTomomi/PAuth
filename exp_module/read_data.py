import numpy as np
import pandas as pd
from pandas import DataFrame

def load_frank_data():
    '''
    expdata.csvの読み込み
    'flag','user2','flag2','user_ave','flag_ave'の削除:107>>102
    '''
    # mainから呼び出すとき(basic)
    # パスの指定は実行するプログラムの相対パスっぽい
    df_ori = pd.read_csv("../10_feature_selection/expdata.csv", sep = ",")
    # 同モジュール内から呼び出すとき
    # df_ori = pd.read_csv("../../10_feature_selection/expdata.csv", sep = ",")
    df_drop = df_ori.drop({'Unnamed: 0', 'flag', 'user2','flag2', 'user_ave', 'flag_ave'}, axis = 1)

    return df_drop

if __name__ == "__main__":
    df_frank = load_frank_data()
    print(df_frank.info())
    print(df_frank.head())
