'''
1: Wikipedia article
2: Wikipedia article
3: Wikipedia article
4: Image comparison game
5: Image comparison game
6: Wikipedia article
7: Image comparison game

doc idごとで実験時間が別れていそうなのでこれも考慮してデータのきりわけ
'''

'''
STEP2 : ドキュメントごとに分割.
後で直すので...
'''


import numpy as np
import pandas as pd
import os

# header = Noneにすることで一行目をヘッダーとして読み込まないそうな
for i in range(1,42):
    df = pd.read_csv(f'fM/fM{i}.csv', header = None, index_col=0)
    df = df.copy().reset_index(drop = True)

    df.columns = ['user', 'doc', 'stroke_inter', 'stroke_duration', 'start_x', 'start_y', 'stop_x', 'stop_y', 'direct_ete_distance', 'mean_result_leng', 'flag', 'direct_ete_line', 'phone', '20_pairwise_v', '50_pairwise_v', '80_pairwise_v', '20_pairwise_acc', '50_pairwise_acc', '80_pairwise_acc', '3ots_m_v', 'ete_larg_deviation', '20_ete_line', '50_ete_line', '80_ete_line', 'ave_direction', 'length_trajectory', 'ratio_ete', 'ave_v', '5points_m_acc', 'm_stroke_press', 'm_stroke_area_cover', 'finger_orien','cd_finger_orien', 'phone_orien' ]

    # st = set{1,2,3,4,5,6,7}
    # ls = df['doc'].value_counts().index.tolist()
    # stli = set(ls)
    # stt = st & stli
    # lss = list(stt)

    ls = df['doc'].value_counts().index.tolist()
    print(f'{i}:{sorted(ls)}')

    os.makedirs('fM_doc', exist_ok = True)
    for j in range(len(ls)):
        df_doc = df[df['doc'] == ls[j]]
        df_doc.to_csv('fM_doc/fM_doc{0}_{1}.csv'.format(i,ls[j]))

    # これは正直要らない
    # df.to_csv('fM_doc/fM_doc{}.csv'.format(i))

print('OK')
