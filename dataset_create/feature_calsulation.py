'''
STEP3 : 特徴量の計算を行う.
後で直すので...
'''
#import
import numpy as np
import pandas as pd
import math
import os

# warning inogre code
import warnings
warnings.filterwarnings('ignore')

'''
orignaldatas read
41 users
'''

def load_frank_data(menber,document,ls):
    df_read = pd.read_csv("fM_doc/fM_doc{0}_{1}.csv".format(menber,ls[document]), sep = ",", index_col = 0)
    df_doc = df_read.copy().reset_index(drop = True)

    # Document_IDと電話番号の削除
    df_drop_doc_phone = df_doc.drop({'doc','phone'}, axis = 1)

    return df_drop_doc_phone, df_drop_doc_phone.drop(df_drop_doc_phone.index[len(df_drop_doc_phone)-1]), df_drop_doc_phone.drop(0)


for menber in range(1,42,1):
    df_read = pd.read_csv(f'fM/fM{menber}.csv', header = None, index_col=0)
    df_ori = df_read.copy().reset_index(drop = True)

    df_ori.columns = ['user', 'doc', 'stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y', 'direct_ete_distance', 'mean_result_leng', 'flag', 'direct_ete_line', 'phone', '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc', '50_pairwise_acc', '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation', '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction', 'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc', 'm_stroke_press', 'm_stroke_area_cover', 'finger_orien','cd_finger_orien', 'phone_orien' ]

    ls = df_ori['doc'].value_counts().index.tolist()

    for document in range(len(ls)):
        # データの読み込み
        # dddp = df_drop_doc_phone
        dddp, dddp_del_lastrow, dddp_del_firstrow = load_frank_data(menber,document,ls)

        dddp_del_firstrow.columns = ['user2', 'stroke_inter2', 'stroke_duration2', 'start_x2', 'start_y2', 'stop_x2', 'stop_y2', 'direct_ete_distance2', 'mean_result_leng2', 'flag2', 'direct_ete_line2', '20_pairwise_v2', '50_pairwise_v2', '80_pairwise_v2', '20_pairwise_acc2', '50_pairwise_acc2', '80_pairwise_acc2', '3ots_m_v2', 'ete_larg_deviation2', '20_ete_line2', '50_ete_line2', '80_ete_line2', 'ave_direction2', 'length_trajectory2', 'ratio_ete2', 'ave_v2', '5points_m_acc2', 'm_stroke_press2', 'm_stroke_area_cover2', 'finger_orien2','cd_finger_orien2', 'phone_orien2' ]

        # 2つのストロークの平均
        b = {'user':{}, 'stroke_inter':{}, 'stroke_duration':{}, 'start_x':{}, 'start_y':{}, 'stop_x':{}, 'stop_y':{}, 'direct_ete_distance':{}, 'mean_result_leng':{}, 'flag':{}, 'direct_ete_line':{}, '20_pairwise_v':{}, '50_pairwise_v':{}, '80_pairwise_v':{}, '20_pairwise_acc':{}, '50_pairwise_acc':{}, '80_pairwise_acc':{}, '3ots_m_v':{}, 'ete_larg_deviation':{}, '20_ete_line':{}, '50_ete_line':{}, '80_ete_line':{}, 'ave_direction':{}, 'length_trajectory':{}, 'ratio_ete':{}, 'ave_v':{}, '5points_m_acc':{}, 'm_stroke_press':{}, 'm_stroke_area_cover':{}, 'finger_orien':{},'cd_finger_orien':{}, 'phone_orien':{}}

        for column_name, item in dddp.iteritems():
            count = 0
            for i in range(len(dddp.index)-1):
                a = (dddp[column_name].iloc[i] + (dddp[column_name].iloc[i+1])) / 2
                b[column_name][count] = a
                count += 1
        dddp_ave = pd.DataFrame(b)

        dddp_ave.columns = ['user_ave', 'stroke_inter_ave', 'stroke_duration_ave', 'start_x_ave', 'start_y_ave', 'stop_x_ave', 'stop_y_ave', 'direct_ete_distance_ave', 'mean_result_leng_ave', 'flag_ave', 'direct_ete_line_ave', '20_pairwise_v_ave', '50_pairwise_v_ave', '80_pairwise_v_ave', '20_pairwise_acc_ave', '50_pairwise_acc_ave', '80_pairwise_acc_ave', '3ots_m_v_ave', 'ete_larg_deviation_ave', '20_ete_line_ave', '50_ete_line_ave', '80_ete_line_ave', 'ave_direction_ave', 'length_trajectory_ave', 'ratio_ete_ave', 'ave_v_ave', '5points_m_acc_ave', 'm_stroke_press_ave', 'm_stroke_area_cover_ave', 'finger_orien_ave','cd_finger_orien_ave', 'phone_orien_ave' ]

        c = {}
        c_lists = ['2stroke_time-','d_stroke_inter-','v_stroke_inter-','a_stroke_inter-', '2stroke_distance-', '2stroke_v-', '2stroke_a-', 'outer_d-','outer_v-', 'outer_a-']

        count2 = 0
        for i in range(len(dddp.index)-1):
            '''
            まさかのstroke_interは1つ目と2つ目のストロークの間の時間じゃなくて，1つ目のストロークの時間＋間の時間だったとさ(おこ)
            '''
             # between1 = dddp['stroke_inter'].iloc[i] - dddp['stroke_duration'].iloc[i])
             # between2 = dddp['stroke_inter'].iloc[i+1] - dddp['stroke_duration'].iloc[i+1])
             # a2 = (dddp['stroke_duration'].iloc[i] + between1 + (dddp['stroke_duration'].iloc[i+1]))
             a2 = (dddp['stroke_duration'].iloc[i] + (dddp['stroke_inter'].iloc[i+1]) + (dddp['stroke_duration'].iloc[i+1]))
             b2 = math.sqrt(pow(((dddp['start_x'].iloc[i+1]) - (dddp['stop_x'].iloc[i])),2) + pow(((dddp['start_y'].iloc[i+1]) - (dddp['stop_y'].iloc[i])),2))
             v = b2 / dddp['stroke_inter'].iloc[i]
             d2 = v / dddp['stroke_inter'].iloc[i]
             e2 = math.sqrt(pow(((dddp['stop_x'].iloc[i]) - (dddp['start_x'].iloc[i])),2) + pow(((dddp['stop_y'].iloc[i]) - (dddp['start_y'].iloc[i])),2)) + b2 + math.sqrt(pow(((dddp['stop_x'].iloc[i+1]) - (dddp['start_x'].iloc[i+1])),2) + pow(((dddp['stop_y'].iloc[i+1]) - (dddp['start_y'].iloc[i+1])),2))
             v2 = e2 / a2
             g2 = v2 / a2
             ee2 = math.sqrt(pow(((dddp['stop_x'].iloc[i+1]) - (dddp['start_x'].iloc[i])),2) + pow(((dddp['stop_y'].iloc[i+1]) - (dddp['start_y'].iloc[i])),2))
             vv2 = ee2 / a2
             gg2 = vv2 / a2

             c[count2] = {'2stroke_time':a2, 'd_stroke_inter':b2, 'v_stroke_inter':v, 'a_stroke_inter':d2, '2stroke_distance':e2, '2stroke_v':v2, '2stroke_a':g2, 'outer_d':ee2, 'outer_v':vv2, 'outer_a':gg2}
             count2 +=1

        dddp_new_features = pd.DataFrame(c)
        # print(df6.head().T)

        df = pd.concat([dddp_del_lastrow.reset_index(drop=True), dddp_del_firstrow.reset_index(drop=True), dddp_ave, dddp_new_features.T], axis=1, join='outer')

        os.makedirs('FC_doc', exist_ok = True)
        df.to_csv('FC_doc/fc{0}_{1}.csv'.format(menber,ls[document]))
