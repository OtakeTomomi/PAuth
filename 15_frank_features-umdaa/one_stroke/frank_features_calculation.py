# データの説明
"""

一部オリジナルと異なる部分があるがもう放置



The columns of the dataset are
'phone ID',
'user ID',
'document ID',
'time[ms]',
'action',
'phone orientation',
'x-coordinate',
'y-coordinate',
'pressure',
'area covered',
'finger orientation'.

Detailed descriptions below.




Phone ID:
indicates the phone and the experimenter that recorded the data
reaches from 1-5
1 : Nexus 1, Experimenter E
2 : Nexus S, Experimenter M
3 : Nexus 1, Experimenter R
4 : Samsung Galaxy S, Experimenter I
5 : Droid Incredible, Experimenter E

user ID:
anonymous users

doc id:
This number indicates the document that the user saw on screen while we collected the data. Every document represents a different session, i.e. the user has put down the device between working on different documents.
The breaks between doc ids 1-5 and 6-7 are several minutes, respectively. Data with doc ids 6 and 7 has been collected 7 to 14 days after doc ids 1-5
1: Wikipedia article
2: Wikipedia article
3: Wikipedia article
4: Image comparison game
5: Image comparison game
6: Wikipedia article
7: Image comparison game


time[ms]:
absolute time of recorded action (ms since 1970).

action:
can take three values 0: touch down, 1: touch up, 2: move finger on screen. In our paper, a stroke is defined as all actions between a 0 and a 1 if there is a xy-displacement between these actions. Clicks are actions between 0 and 1 without displacement.


'phone orientation', 'x-coordinate', 'y-coordinate', 'pressure', 'area covered', 'finger orientation'
are the values returned from the Android API at the current action.



% raw feature columns
col_phoneID = 1;
col_user = 2;
col_doc  = 3;
col_time = 4;
col_act  = 5;
col_x = 7;
col_y = 8;
col_orient = 6;
col_press = 9;
col_area = 10;
col_Forient = 11;


% number of extracted features
Nfeat = 34;


% feature descriptors
featureStr{1} ='user id';
featureStr{2  } = 'doc id';
featureStr{3  } = 'inter-stroke time';
featureStr{4  } = 'stroke duration';
featureStr{5  } = 'start $x$';
featureStr{6  } = 'start $y$';
featureStr{7  } = 'stop $x$';
featureStr{8} = 'stop $y$';
featureStr{9} = 'direct end-to-end distance';
featureStr{10} = ' mean resultant lenght';
featureStr{11} = 'up/down/left/right flag';
featureStr{12} = 'direction of end-to-end line';
featureStr{13} = 'phone id';
featureStr{14} = '20\%-perc. pairwise velocity';→怪しい
featureStr{15} = '50\%-perc. pairwise velocity';
featureStr{16} = '80\%-perc. pairwise velocity';→怪しい
featureStr{17} = '20\%-perc. pairwise acc';→怪しい
featureStr{18} = '50\%-perc. pairwise acc';
featureStr{19} = '80\%-perc. pairwise acc';→怪しい
featureStr{20 } = 'median velocity at last 3 pts';→一部怪しい
featureStr{21 } = 'largest deviation from end-to-end line';→怪しい
featureStr{22} = '20\%-perc. dev. from end-to-end line';→怪しい
featureStr{23} = '50\%-perc. dev. from end-to-end line';→怪しい
featureStr{24} = '80\%-perc. dev. from end-to-end line';→怪しい
featureStr{25  } = 'average direction';
featureStr{26 } = 'length of trajectory';
featureStr{27 } = 'ratio end-to-end dist and length of trajectory';
featureStr{28 } = 'average velocity';
featureStr{29} = 'median acceleration at first 5 points';
featureStr{30 } = 'mid-stroke pressure';
featureStr{31 } = 'mid-stroke area covered';
featureStr{32 } = 'mid-stroke finger orientation';
featureStr{33} = 'change of finger orientation';
featureStr{34} = 'phone orientation';


"""


import pandas as pd
# import os
import numpy as np
import pingouin as pg
from scipy import signal
import math
# import statistics
# from scipy import stats


from tqdm import tqdm
import time


# warning ignore code
import warnings
warnings.filterwarnings('ignore')

# 表示範囲設定 notebook使用時のみ
# pd.set_option('display.max_rows', 10000)
# pd.set_option('display.max_columns', 100)


def features_process_feank_data(df_frank_orignal_data):
    df_test_copy = df_frank_orignal_data.copy()

    # convert ms to s
    df_test_copy.loc[:, 'col_time'] = df_test_copy.loc[:, 'col_time'] / 1000

    # flip sign of y_axis
    df_test_copy.loc[:, 'col_y'] = -df_test_copy.loc[:, 'col_y']

    # start counting from 1
    df_test_copy.loc[:, 'col_phoneID'] = df_test_copy.loc[:, 'col_phoneID'] + 1
    df_test_copy.loc[:, 'col_user'] = df_test_copy.loc[:, 'col_user'] + 1
    df_test_copy.loc[:, 'col_doc'] = df_test_copy.loc[:, 'col_doc'] + 1

    # find beginning of strokes
    downInd = [i for i, x in enumerate(list(df_test_copy['col_act'])) if x == 0]

    # number of strokes
    Nstrokes = len(downInd)
    # downInd = [downInd; size(t, 1)]; # よくわからん

    # global statistics
    indivPrctlVals = [20, 50, 80]

    Nfeat = 35
    featMat = np.zeros([Nstrokes, Nfeat])
    featMat[:, :] = np.nan

    print(f'extracting features of  {Nstrokes}  strokes..') # 21174

    i = 0
    # Nstrokes-1
    for i in tqdm(range(Nstrokes-1)):
        #         try:
        # downIndはtouch-downしたタイミングのindex
        x_stroke = df_test_copy.loc[downInd[i]:downInd[i + 1] - 1, :]

        # number of measurements of stroke
        npoints = len(x_stroke)

        # featMat.mとカラム番号をあわせるためのダミー列　
        featMat[i, 0] = 0

        # user id
        featMat[i, 1] = x_stroke.iloc[0, 1]
        # doc id
        featMat[i, 2] = x_stroke.iloc[0, 2]
        # phone id
        featMat[i, 13] = x_stroke.iloc[0, 0]

        # time to next stroke (0 if last stroke in dataset)
        if (featMat[i, 3] == 0) or (df_test_copy.loc[downInd[i + 1], 'col_user'] != featMat[i, 1]):
            featMat[i, 3] = 'NaN'
            print(df_test_copy.loc[downInd[i + 1], 'col_user'])
            print(featMat[i, 1])
        else:
            # col_time: 3
            featMat[i, 3] = df_test_copy.loc[min(downInd[Nstrokes - 1], downInd[i + 1]), 'col_time'] - x_stroke.iloc[
                0, 3]

        # time to last point of this stroke
        featMat[i, 4] = x_stroke.iloc[-1, 3] - x_stroke.iloc[0, 3]

        # convert from pixels to mm
        if x_stroke.iloc[0, 0] == 1:
            x_stroke.loc[:, 'col_x'] = (1 / 252) * x_stroke.loc[:, 'col_x'] * 25.4
            x_stroke.loc[:, 'col_y'] = (1 / 252) * x_stroke.loc[:, 'col_y'] * 25.4
        elif x_stroke.iloc[0, 0] == 2:
            x_stroke.loc[:, 'col_x'] = (1 / 233) * x_stroke.loc[:, 'col_x'] * 25.4
            x_stroke.loc[:, 'col_y'] = (1 / 233) * x_stroke.loc[:, 'col_y'] * 25.4
        elif x_stroke.iloc[0, 0] == 3:
            x_stroke.loc[:, 'col_x'] = (1 / 252) * x_stroke.loc[:, 'col_x'] * 25.4
            x_stroke.loc[:, 'col_y'] = (1 / 252) * x_stroke.loc[:, 'col_y'] * 25.4
        elif x_stroke.iloc[0, 0] == 4:
            x_stroke.loc[:, 'col_x'] = (1 / 233) * x_stroke.loc[:, 'col_x'] * 25.4
            x_stroke.loc[:, 'col_y'] = (1 / 233) * x_stroke.loc[:, 'col_y'] * 25.4
        elif x_stroke.iloc[0, 0] == 5:
            x_stroke.loc[:, 'col_x'] = (1 / 252) * x_stroke.loc[:, 'col_x'] * 25.4
            x_stroke.loc[:, 'col_y'] = (1 / 252) * x_stroke.loc[:, 'col_y'] * 25.4

        # col_x = 6, col_y = 7
        # x-pos start
        featMat[i, 5] = x_stroke.iloc[0, 6]
        # y-pos start
        featMat[i, 6] = x_stroke.iloc[0, 7]
        # x-pos end
        featMat[i, 7] = x_stroke.iloc[-1, 6]
        # y-pos start
        featMat[i, 8] = x_stroke.iloc[-1, 7]

        featMat[i, 9] = np.sqrt((featMat[i, 8] - featMat[i, 6]) ** 2 + (featMat[i, 7] - featMat[i, 5]) ** 2)

        # x-displacement : X-変位
        xdispl = signal.lfilter([1, -1], 1, list(x_stroke.iloc[:, 6]))
        xdispl = np.delete(xdispl, 0)

        # y-displacement : Y-変位
        ydispl = signal.lfilter([1, -1], 1, list(x_stroke.iloc[:, 7]))
        ydispl = np.delete(ydispl, 0)

        # pairwise time diffs : ペアワイズ時間の差分
        tdelta = signal.lfilter([1, -1], 1, list(x_stroke.iloc[:, 3]))
        tdelta = np.delete(tdelta, 0)

        # pairw angle　: ペアの角度
        angl = np.arctan2(ydispl, xdispl)

        # Mean Resutlant Length (requires circular statistics toolbox)
        try:
            featMat[i, 10] = pg.circ_r(angl)
        except Exception as e:
            # print(f' error row: {i} {e}')
            pass

        # pairwise displacements : ペアワイズ変位
        # C = sqrt(abs(A).^2 + abs(B).^2) は hypot(A, B)と同じ（matlab）
        # pairwDist = sqrt(xdispl.^2 + ydispl.^2);
        pairwDist = [math.hypot(item_x, item_y) for item_x, item_y in zip(xdispl.tolist(), ydispl.tolist())]

        # speed histogram : 速度ヒストグラム
        try:
            v = pairwDist / tdelta
            featMat[i, 14:17] = np.percentile(v, indivPrctlVals, interpolation='midpoint')
        except Exception as e:
            # print(f' error row: {i} {e}')
            pass

            # full stat stuff
        try:
            # acceleration histogram : 加速度ヒストグラム
            a = signal.lfilter([1, -1], 1, v)
            a = a / tdelta
            a = np.delete(a, 0)

            featMat[i, 17:20] = np.percentile(a, indivPrctlVals, interpolation='midpoint')
        except Exception as e:
            # print(f' error row: {i} {e}')
            pass
        # median velocity of last 3 points : 最後の3点の中央値速度
        try:
            v_list = v.tolist()
            v_list_last3_sort = sorted([v_list[len(v_list) - 3], v_list[len(v_list) - 2], v_list[len(v_list) - 1]])
            featMat[i, 20] = np.median(v_list_last3_sort[1:3])
        except Exception as e:
            # print(f' error row: {i} {e}')
            pass
        #             featMat[i, 20] = statistics.median(v_list[v_list.index(max(v_list[len(v_list)-3],v_list[len(v_list)-2], v_list[len(v_list)-1])): v_list.index(v_list[len(v_list)-1])])
        #             featMat[i, 21] = 0

        # max dist. beween direct line and true line (with sign):直線と真線の最大距離（符号あり
        xvek = x_stroke.iloc[:, 6] - x_stroke.iloc[0, 6]
        yvek = x_stroke.iloc[:, 7] - x_stroke.iloc[0, 7]

        # project each vector on straight line : 各ベクトルを直線上に投影
        # compute unit line perpendicular to straight connection and project on this
        # 直線接続に垂直な単位線を計算して、投影
        perVek = np.cross([xvek.iloc[-1], yvek.iloc[-1], 0], [0, 0, 1])
        perVek = perVek / np.sqrt(np.array([perVek[0] * perVek[0], perVek[1] * perVek[1], 0]))
        perVek_num = [0 if x == False else 1 for x in np.isnan(perVek).tolist()]

        # happens if vectors have length 0
        # ベクトルが長さ 0 の場合
        if np.all(np.array(perVek_num) == 0):
            perVek = 0

        # all distances to direct line:直線までの全距離
        lst_xvex = [perVek[0] for i in range(len(xvek))]
        lst_yvex = [perVek[1] for i in range(len(xvek))]

        projectOnPerpStraight = (xvek * lst_xvex) + (yvek * lst_yvex)

        # report maximal (absolute) distance: 最大(絶対)距離を報告
        # https://datachemeng.com/matlab_to_python/
        absProj = abs(projectOnPerpStraight)
        #         maxind = find( absProj == max( absProj ) )
        featMat[i, 21] = max(absProj)

        # stat of distances (bins are not the same for all strokes):距離の統計量 (ビンはすべてのストロークで同じではない)
        featMat[i, 22:25] = np.percentile(projectOnPerpStraight, indivPrctlVals, interpolation='midpoint')

        # average direction of ensemble of pairs: ペアのアンサンブルの平均方向
        try:
            featMat[i, 25] = pg.circ_mean(angl)
        except Exception as e:
            # print(f' error row: {i} {e}')
            pass

        # direction of end-to-end line: 端から端までの直線の方向
        featMat[i, 12] = np.arctan2((featMat[i, 8] - featMat[i, 6]), (featMat[i, 7] - featMat[i, 5]))

        # direction flag 1 up, 2 down, 3 left 4 right  (see doc atan2): in what direction is screen being moved to?
        # 方向フラグ 1 上向き，2 下向き，3 左向き，4 右向き:画面はどの方向に移動しているか

        # convert to [0 2pi]
        tmpangle = featMat[i, 12] + math.pi
        if tmpangle <= (math.pi / 4):
            # right
            featMat[i, 11] = 4
        elif (tmpangle > math.pi / 4) and (tmpangle <= 5 * math.pi / 4):
            # up or left
            if tmpangle < 3 * math.pi / 4:
                # up
                featMat[i, 11] = 1
            else:
                # left
                featMat[i, 11] = 2
        else:
            # down or right
            if tmpangle < 7 * math.pi / 4:
                # down
                featMat[i, 11] = 3
            else:
                # right
                featMat[i, 11] = 4

        # length of trajectory : 軌跡の長さ
        featMat[i, 26] = sum(pairwDist)

        # ratio between direct length and length of trajectory : 直接の長さと軌道の長さの比
        featMat[i, 27] = featMat[i, 9] / featMat[i, 26]

        # average velocity : 平均速度
        featMat[i, 28] = featMat[i, 26] / featMat[i, 4]

        # average acc over first 5 points : 最初の 5 ポイントの平均アクセント
        featMat[i, 29] = np.median(a[0:min(5, len(a))])

        # pressure in the middle of the stroke: ストロークの途中の圧力
        # col_press = 8
        col_press_stroke = x_stroke['col_press'].values.tolist()
        featMat[i, 30] = np.median(col_press_stroke[int(np.floor(npoints / 2)) - 1:int(np.ceil(npoints / 2))])

        # covered area in the middle of the stroke
        # col_area = 9
        col_area_stroke = x_stroke['col_area'].values.tolist()
        featMat[i, 31] = np.median(col_area_stroke[int(np.floor(npoints / 2)) - 1:int(np.ceil(npoints / 2))])

        # finger orientation in the middle of the stroke
        col_forient_stroke = x_stroke['col_forient'].values.tolist()
        featMat[i, 32] = np.median(col_forient_stroke[int(np.floor(npoints / 2)) - 1:int(np.ceil(npoints / 2))])

        # change of finger orientation during stroke
        # 'col_forient' = 10
        featMat[i, 33] = x_stroke.iloc[len(x_stroke) - 1, 10] - x_stroke.iloc[0, 10]

        # phone orientation
        # col_orient = 5
        featMat[i, 34] = x_stroke.iloc[0, 5]

    return featMat


if __name__ == '__main__':
    print("This is frank original data to features calculation data ")

    # データの読み込み
    df_ori = pd.read_csv('../../01_data/data.csv',
                         names=['col_phoneID', 'col_user', 'col_doc', 'col_time', 'col_act', 'col_orient', 'col_x',
                                'col_y', 'col_press', 'col_area', 'col_forient'])
    # df_ori.head()
    df_ori_copy = df_ori.copy()

    featMat = features_process_feank_data(df_ori_copy)

    df_featMat = pd.DataFrame(featMat)

    df_featMat = df_featMat.drop(0, axis=1)

    print(df_featMat)

    df_featMat.to_csv('frank_features_calc.csv', header=False, index=False)

