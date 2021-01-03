"""
This module read datasets.
The current dataset to be loaded is a frank dataset only.
"""

# import os.path
import os
import pandas as pd
import pprint
# import pickle


def read_frank(combination_flag):
    if combination_flag:
        df_ori_t = pd.read_csv('/Users/otaketomomi/PycharmProjects/PAuth/dataset_create/expdata_doc.csv', sep=',')
        # 不要なものを列で削除101
        # df_drop_t = df_ori_t.drop({'Unnamed: 0', 'flag', 'user2', 'doc2', 'flag2', 'user_ave', 'doc_ave', 'flag_ave'},
        #                           axis=1)

        # 'finger_orien', 'cd_finger_orien', 'phone_orien'を削除する場合91
        df_drop_t = df_ori_t.drop({'Unnamed: 0', 'flag', 'finger_orien', 'cd_finger_orien', 'phone_orien',
                                   'user2', 'doc2', 'flag2', 'finger_orien2', 'cd_finger_orien2', 'phone_orien2',
                                   'user_ave', 'doc_ave', 'flag_ave', 'finger_orien_ave', 'cd_finger_orien_ave',
                                   'phone_orien_ave'}, axis=1)

        # 平均したものを削除
        df_drop_t2 = df_drop_t.drop({'stroke_inter_ave', 'stroke_duration_ave', 'start_x_ave', 'start_y_ave',
                                     'stop_x_ave', 'stop_y_ave', 'direct_ete_distance_ave', 'mean_result_leng_ave',
                                     'direct_ete_line_ave', '20_pairwise_v_ave', '50_pairwise_v_ave', '80_pairwise_v_ave',
                                     '20_pairwise_acc_ave', '50_pairwise_acc_ave', '80_pairwise_acc_ave', '3ots_m_v_ave',
                                     'ete_larg_deviation_ave', '20_ete_line_ave', '50_ete_line_ave', '80_ete_line_ave',
                                     'ave_direction_ave', 'length_trajectory_ave', 'ratio_ete_ave', 'ave_v_ave',
                                     '5points_m_acc_ave', 'm_stroke_press_ave', 'm_stroke_area_cover_ave'}, axis=1)

        return df_drop_t2
    else:
        df_ori_f = pd.read_csv("/Users/otaketomomi/PycharmProjects/PAuth/02_features/featMat.csv", sep=",")
        df_ori_f.columns = ['user', 'doc', 'stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y',
                            'direct_ete_distance', 'mean_result_leng', 'flag', 'direct_ete_line', 'phone',
                            '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc', '50_pairwise_acc',
                            '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation', '20_ete_line', '50_ete_line',
                            '80_ete_line', 'ave_direction', 'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc',
                            'm_stroke_press', 'm_stroke_area_cover', 'finger_orien', 'cd_finger_orien', 'phone_orien']
        # df_drop = df_ori_f.drop('phone', axis=1)
        # 'finger_orien', 'cd_finger_orien', 'phone_orien'を削除する場合28
        df_drop = df_ori_f.drop({'phone', 'finger_orien', 'cd_finger_orien', 'phone_orien'}, axis=1)
        df_drop_f = df_drop.dropna(axis=0, how='any')
        return df_drop_f


def load_frank(combination=True):
    """frank datasetの読み込み
    Parameters
    combination :
        Trueの場合は2つ組み合わせたデータセット
        Falseの場合は通常のもの
    return df_t or df_f
    """
    if combination:
        df_t = read_frank(True)
        return df_t
    else:
        df_f = read_frank(False)
        return df_f


if __name__ == '__main__':
    print("This is module! The name is dataset ")

    print('\n combination=True')
    df1 = load_frank(True)
    print(df1.head())
    pprint.pprint(df1.shape)
    print(df1.columns)

    print('\n combination=False')
    df2 = load_frank(False)
    print(df2.head())
    print(df2.shape)
    print(df2.columns)
