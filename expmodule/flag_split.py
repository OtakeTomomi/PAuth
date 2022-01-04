"""
This module split dataset by Stroke direction.

変換表
a = 1:up
b = 2:left
c = 3:down
d = 4:right
"""



def flag4(df, flagtext):
    """
    molti_flag列の11~44のflagの値ごとにgroupbyを行いまとめる
    flagにmulti_flagの値(key)
    sdfに(value)
    groupbyはイテレーション用のメソッドを持っているのでこれでkeyに基づいてデータを分割できる
    """
    df2 = df.copy()
    for flag, sdf in df2.groupby(flagtext):
        sdf = sdf.reset_index(drop=True)
        if flag == 1:
            re_a = sdf.drop(flagtext, axis=1)
            # print(df_up.groupby("user").size()) #個数を調べる
        elif flag == 2:
            re_b = sdf.drop(flagtext, axis=1)
        elif flag == 3:
            re_c = sdf.drop(flagtext, axis=1)
        else:
            re_d = sdf.drop(flagtext, axis=1)
    # flagごとのデータを返す
    return re_a, re_b, re_c, re_d


def flag16(df, flagtext):
    df2 = df.copy()
    for flag, sdf in df2.groupby(flagtext):
        sdf = sdf.reset_index(drop=True)
        if flag == 11:
            re_aa = sdf.drop(flagtext, axis=1)
        elif flag == 12:
            re_ab = sdf.drop(flagtext, axis=1)
        elif flag == 13:
            re_ac = sdf.drop(flagtext, axis=1)
        elif flag == 14:
            re_ad = sdf.drop(flagtext, axis=1)
        elif flag == 21:
            re_ba = sdf.drop(flagtext, axis=1)
        elif flag == 22:
            re_bb = sdf.drop(flagtext, axis=1)
        elif flag == 23:
            re_bc = sdf.drop(flagtext, axis=1)
        elif flag == 24:
            re_bd = sdf.drop(flagtext, axis=1)
        elif flag == 31:
            re_ca = sdf.drop(flagtext, axis=1)
        elif flag == 32:
            re_cb = sdf.drop(flagtext, axis=1)
        elif flag == 33:
            re_cc = sdf.drop(flagtext, axis=1)
        elif flag == 34:
            re_cd = sdf.drop(flagtext, axis=1)
        elif flag == 41:
            re_da = sdf.drop(flagtext, axis=1)
        elif flag == 42:
            re_db = sdf.drop(flagtext, axis=1)
        elif flag == 43:
            re_dc = sdf.drop(flagtext, axis=1)
        elif flag == 44:
            re_dd = sdf.drop(flagtext, axis=1)
    # flagごとのデータを返す
    return re_aa, re_ab, re_ac, re_ad, re_ba, re_bb, re_bc, re_bd, re_ca, re_cb, re_cc, re_cd, re_da, re_db, re_dc, re_dd


if __name__ == '__main__':
    print("This is module! The name is flag_split ")
