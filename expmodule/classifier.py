"""

"""
import pandas as pd


# 多クラスの場合
def user_select(df_select, select_menber):
    """実験に使用するユーザの選択とソート
    :param df_select: 上下左右方向で分けられたデータ(all users)
    :param select_menber: all usersから何人実験に使用するか
    :return: select_menberで指定された人数を抽出したデータ(データの多い順)
    """

    # フラグ内のuserごとにデータをまとめる
    df_flag_user = df_select.groupby("user")
    # 降順にフラグの数でソート
    df_flag_user_sort = pd.DataFrame(df_flag_user.size().sort_values(ascending=False))
    # データ数の多いuserからリストに格納
    user_max = df_flag_user_sort.index.values

    df_flag_select = df_select[df_select['user'].isin(user_max[:select_menber])]

    # ff = df_flag_select.groupby("user")
    # print(ff.size().sort_values(ascending=False))
    # print("選択されているもの→　メンバー：{}人".format(select_menber))
    # print("=========================")
    return df_flag_select


# 多クラスの場合
# def data_select(df_user, data_number=int('all')):
#

if __name__ == '__main__':
    print("This is module! The name is classifier ")
