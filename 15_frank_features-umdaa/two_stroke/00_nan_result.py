'''
欠損値確認
nan_result.csv作成プログラム.
'''
#import
import numpy as np
import pandas as pd


# warning inogre code
import warnings
warnings.filterwarnings('ignore')

'''
orignaldatas read
41 users
'''

nan_result = pd.DataFrame(columns = ['user', 'doc', 'stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y', 'direct_ete_distance', 'mean_result_leng', 'flag', 'direct_ete_line', 'phone', '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc', '50_pairwise_acc', '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation', '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction', 'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc', 'm_stroke_press', 'm_stroke_area_cover', 'finger_orien','cd_finger_orien', 'phone_orien' ])

for menber in range(1,42,1):
    def load_frank_data(i):
        df2 = pd.read_csv("featMat_fu/featMat_fu{}.csv".format(menber), sep = ",", header = None, index_col = 0)
        df = df2.copy().reset_index(drop = True)

        # 34 columns name update
        df.columns = ['user', 'doc', 'stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y', 'direct_ete_distance', 'mean_result_leng', 'flag', 'direct_ete_line', 'phone', '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc', '50_pairwise_acc', '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation', '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction', 'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc', 'm_stroke_press', 'm_stroke_area_cover', 'finger_orien','cd_finger_orien', 'phone_orien' ]

        return df
    df = load_frank_data(menber)
    nan_result = nan_result.append(df.isnull().sum(), ignore_index=True)
    nan_result.index = range(1, nan_result.shape[0]+1)

nan_result.T.to_csv('nan_result.csv')