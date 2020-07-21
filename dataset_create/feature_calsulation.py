"""
STEP3 : 特徴量の計算を行う.
"""

import pandas as pd
import math
import os

# warning inogre code
import warnings
warnings.filterwarnings('ignore')

"""
orignaldatas read
41 users
"""

# データの読み込み
def load_frank_data(user_n, document, doc_list):
    """
    :param user_n: userの識別番号
    :param document: ドキュメント番号
    :param doc_list:
    :return: 1つ目は完全，2つ目は1つ目のデータの最終行を削除したもの，3つ目は1つ目のデータの最初の行を削除したもの
    """
    # fM_docから各ユーザの各ドキュメントを読み込み
    df_read_fM_doc = pd.read_csv("fM_doc/fM_doc{0}_{1}.csv".format(user_n, doc_list[document]), sep=",", index_col=0)
    # indexのリセット
    df_doc = df_read_fM_doc.copy()
    df_doc = df_doc.reset_index(drop=True)

    # Document_IDと電話番号の削除
    df_drop_doc_phone = df_doc.drop({'doc', 'phone'}, axis=1)

    return df_drop_doc_phone, df_drop_doc_phone.drop(df_drop_doc_phone.index[len(df_drop_doc_phone)-1]),\
           df_drop_doc_phone.drop(0)


for user_n in range(1, 42, 1):
    # fMよりuser_nに該当するuserのデータを読み込み
    # doc_listを作成するためだけっぽいな
    df_read_fM = pd.read_csv(f'fM/fM{user_n}.csv', header=None, index_col=0)
    df_ori = df_read_fM.copy().reset_index(drop=True)

    df_ori.columns = ['user', 'doc', 'stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y',
                      'direct_ete_distance', 'mean_result_leng', 'flag', 'direct_ete_line', 'phone', '20_pairwise_v',
                      '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc', '50_pairwise_acc', '80_pairwise_acc',
                      '3ots_m_v', 'ete_larg_deviation', '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction',
                      'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc', 'm_stroke_press',
                      'm_stroke_area_cover', 'finger_orien', 'cd_finger_orien', 'phone_orien']

    doc_list = df_ori['doc'].value_counts().index.tolist()
    # 重要？本番はここから
    for document in range(len(doc_list)):
        # データの読み込み
        # dddp = df_drop_doc_phone
        dddp, dddp_del_lastrow, dddp_del_firstrow = load_frank_data(user_n, document, doc_list)

        # dddp_del_firstrowのColumnsを書き換え
        dddp_del_firstrow.columns = ['user2', 'stroke_inter2', 'stroke_duration2', 'start_x2', 'start_y2', 'stop_x2',
                                     'stop_y2', 'direct_ete_distance2', 'mean_result_leng2', 'flag2',
                                     'direct_ete_line2', '20_pairwise_v2', '50_pairwise_v2', '80_pairwise_v2',
                                     '20_pairwise_acc2', '50_pairwise_acc2', '80_pairwise_acc2', '3ots_m_v2',
                                     'ete_larg_deviation2', '20_ete_line2', '50_ete_line2', '80_ete_line2',
                                     'ave_direction2', 'length_trajectory2', 'ratio_ete2',
                                     'ave_v2', '5points_m_acc2', 'm_stroke_press2',
                                     'm_stroke_area_cover2', 'finger_orien2', 'cd_finger_orien2', 'phone_orien2']

        # dddp_del_lastrowとdddp_del_firstlowのストロークの平均を格納する辞書の用意
        dddp_fl_mean = {'user': {}, 'stroke_inter': {}, 'stroke_duration': {}, 'start_x': {}, 'start_y': {},
                        'stop_x': {}, 'stop_y': {}, 'direct_ete_distance': {}, 'mean_result_leng': {}, 'flag': {},
                        'direct_ete_line': {}, '20_pairwise_v': {}, '50_pairwise_v': {}, '80_pairwise_v': {},
                        '20_pairwise_acc': {}, '50_pairwise_acc': {}, '80_pairwise_acc': {}, '3ots_m_v': {},
                        'ete_larg_deviation': {}, '20_ete_line': {}, '50_ete_line': {},
                        '80_ete_line': {}, 'ave_direction': {}, 'length_trajectory': {}, 'ratio_ete': {}, 'ave_v': {},
                        '5points_m_acc': {}, 'm_stroke_press': {}, 'm_stroke_area_cover': {}, 'finger_orien': {},
                        'cd_finger_orien': {}, 'phone_orien': {}}

        # 辞書型のループ，columns_name=key, item=value
        for column_name, item in dddp.iteritems():
            count = 0
            # dddpの一番最期から2番目までを繰り返す
            for i in range(len(dddp.index)-1):
                a = (dddp[column_name].iloc[i] + (dddp[column_name].iloc[i+1])) / 2
                dddp_fl_mean[column_name][count] = a
                count += 1
        # データフレームに変換
        dddp_ave = pd.DataFrame(dddp_fl_mean)
        # dddp_aveのColumnsを書き換え
        dddp_ave.columns = ['user_ave', 'stroke_inter_ave', 'stroke_duration_ave', 'start_x_ave', 'start_y_ave',
                            'stop_x_ave', 'stop_y_ave', 'direct_ete_distance_ave', 'mean_result_leng_ave', 'flag_ave',
                            'direct_ete_line_ave', '20_pairwise_v_ave', '50_pairwise_v_ave', '80_pairwise_v_ave',
                            '20_pairwise_acc_ave', '50_pairwise_acc_ave', '80_pairwise_acc_ave', '3ots_m_v_ave',
                            'ete_larg_deviation_ave', '20_ete_line_ave', '50_ete_line_ave', '80_ete_line_ave',
                            'ave_direction_ave', 'length_trajectory_ave', 'ratio_ete_ave', 'ave_v_ave',
                            '5points_m_acc_ave', 'm_stroke_press_ave', 'm_stroke_area_cover_ave', 'finger_orien_ave',
                            'cd_finger_orien_ave', 'phone_orien_ave']
        # 追加する新しい特徴を保存する枠
        add_features = {}
        # これは直接は使用しないけど内訳を確認したい時用
        add_features_lists = ['2stroke_time-', 'd_stroke_inter-', 'v_stroke_inter-', 'a_stroke_inter-',
                              '2stroke_distance-', '2stroke_v-', '2stroke_a-', 'outside_d-', 'outside_v-', 'outside_a-']
        count2 = 0
        for i in range(len(dddp.index)-1):
            # まさかのstroke_interは1つ目と2つ目のストロークの間の時間じゃなくて，1つ目のストロークの時間＋間の時間だった
            # 1つ目のストロークと2つ目のストロークの間の時間
            between1 = dddp['stroke_inter'].iloc[i] - dddp['stroke_duration'].iloc[i]
            # 2つ目のストロークと3つ目のストロークの間の時間
            between2 = dddp['stroke_inter'].iloc[i] - dddp['stroke_duration'].iloc[i]

            # 合計ストローク時間
            D_total = (dddp['stroke_duration'].iloc[i] + between1 + (dddp['stroke_duration'].iloc[i+1]))
            # a2 = (dddp['stroke_duration'].iloc[i] +
            # (dddp['stroke_inter'].iloc[i + 1]) + (dddp['stroke_duration'].iloc[i + 1]))

            # ストローク間の距離
            d_inter = (math.sqrt(pow(((dddp['start_x'].iloc[i+1]) - (dddp['stop_x'].iloc[i])), 2) +
                                 pow(((dddp['start_y'].iloc[i+1]) - (dddp['stop_y'].iloc[i])), 2)))
            # ストローク間の速度
            v_inter = d_inter / between1

            # ストローク間の加速度
            a_inter = v_inter / between1

            # 2つのストロークの合計の距離(長さ)
            d_sum = (math.sqrt(pow(((dddp['stop_x'].iloc[i]) - (dddp['start_x'].iloc[i])), 2) +
                               pow(((dddp['stop_y'].iloc[i]) - (dddp['start_y'].iloc[i])), 2)) +
                     d_inter +
                     math.sqrt(pow(((dddp['stop_x'].iloc[i+1]) - (dddp['start_x'].iloc[i+1])), 2) +
                               pow(((dddp['stop_y'].iloc[i+1]) - (dddp['start_y'].iloc[i+1])), 2)))
            # 2つのストロークにおける速度
            v_outer = d_sum / D_total

            # 2つのストロークにおける加速度
            a_outer = v_outer / D_total

            # 2つのストロークの開始点と終了点の距離
            d_outside = (math.sqrt(pow(((dddp['stop_x'].iloc[i+1]) - (dddp['start_x'].iloc[i])), 2) +
                                   pow(((dddp['stop_y'].iloc[i+1]) - (dddp['start_y'].iloc[i])), 2)))
            # 2つのストロークの開始点と終了点の距離における速度
            v_outside = d_outside / D_total

            # 2つのストロークの開始点と終了点の距離における加速度
            a_outside = v_outside / D_total

            add_features[count2] = {'2stroke_time': D_total, 'd_stroke_inter': d_inter, 'v_stroke_inter': v_inter,
                                    'a_stroke_inter': a_inter, '2stroke_distance': d_sum,
                                    '2stroke_v': v_outer, '2stroke_a': a_outer, 'outside_d': d_outside,
                                    'outside_v': v_outside, 'outside_a': a_outside}
            count2 += 1

        # 新しく算出した特徴量をデータフレームに変換
        dddp_new_features = pd.DataFrame(add_features)
        # print(df6.head().T)
        # データの結合，1つ目のストローク＋2つ目のストローク＋2つのストロークの平均＋新しく算出した特徴量の順番
        df = pd.concat([dddp_del_lastrow.reset_index(drop=True), dddp_del_firstrow.reset_index(drop=True), dddp_ave,
                        dddp_new_features.T], axis=1, join='outer')
        # FC_docファイルがなければ作成
        os.makedirs('FC_doc', exist_ok=True)
        # 各ユーザの各ドキュメントごとに書き出しを行う
        df.to_csv('FC_doc/fc{0}_{1}.csv'.format(user_n, doc_list[document]))
