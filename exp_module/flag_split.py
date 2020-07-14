'''
multi_flagをもとにデータを分割する関数
if文多すぎて恐怖なんだが
変換表
a = 1:up
b = 2:right
c = 3:down
d = 4:left
'''

# import pandas as pd
# from pandas import DataFrame

# frankdatasetにおけるflag_splitの略でfrank_fs
def frank_fs(frank_df):
    df = frank_df
    '''
    molti_flag列の11~44のflagの値ごとにgroupbyを行いまとめる
    flagにmulti_flagの値(key) 
    sdfに(value)
    groupbyはイテレーション用のメソッドを持っているのでこれでkeyに基づいてデータを分割できる
    '''
    for flag, sdf in df.groupby('multi_flag'):
        sdf = sdf.reset_index(drop = True)
        if flag == 11:
            aa = sdf.drop('multi_flag', axis=1)
        elif flag == 12 :
            ab = sdf.drop('multi_flag', axis=1)
        elif flag == 13:
            ac = sdf.drop('multi_flag', axis=1)
        elif flag == 14:
            ad = sdf.drop('multi_flag', axis=1)
        elif flag == 21:
            ba = sdf.drop('multi_flag', axis=1)
        elif flag == 22 :
            bb = sdf.drop('multi_flag', axis=1)
        elif flag == 23:
            bc = sdf.drop('multi_flag', axis=1)
        elif flag == 24:
            bd = sdf.drop('multi_flag', axis=1)
        elif flag == 31:
            ca = sdf.drop('multi_flag', axis=1)
        elif flag == 32 :
            cb = sdf.drop('multi_flag', axis=1)
        elif flag == 33:
            cc = sdf.drop('multi_flag', axis=1)
        elif flag == 34:
            cd = sdf.drop('multi_flag', axis=1)
        elif flag == 41:
            da = sdf.drop('multi_flag', axis=1)
        elif flag == 42 :
            db = sdf.drop('multi_flag', axis=1)
        elif flag == 43:
            dc = sdf.drop('multi_flag', axis=1)
        elif flag == 44:
            dd = sdf.drop('multi_flag', axis=1)
    # flagごとのデータを返す
    return aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd

if __name__ == "__main__":
    # データの読み込み
    import read_data
    frank_df = read_data.load_frank_data()
    aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = frank_fs(frank_df)
