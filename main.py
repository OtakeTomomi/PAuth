'''
メインの実験プログラムのつもり
条件：2ストロークの組み合わせ，分類器は1クラス分類器使用.
'''

# basic
import os
import copy
import pickle
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
# from IPython.display import display

# モデル
import sklearn
from sklearn import svm
from sklearn.svm import OneClassSVM
# from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

#スケーリング
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# その他
# from tqdm import tqdm_notebook as tqdm
import time
from tqdm import tqdm
from multiprocessing import cpu_count
# from sklearn.externals import joblib


# warning inogre code
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
from exp_module import read_data as rd
frank_df = rd.load_frank_data()

# データをmulti_flagを基準に分割
# a,b,c,dのストローク方向はflag_splitに記載
from exp_module import flag_split
aa, ab, ac, ad, ba, bb, bc, bd, ca, cb, cc, cd, da, db, dc, dd = flag_split.frank_fs(frank_df)
# 各multi_flagに含まれる各ユーザのデータ数の多い順について確認したい場合にはlist_index = conform.conf_sel_flag_qty()で確認可能ではある

# 選択されたユーザのデータを各aa~ddから抽出
# select_user_from_frank_fs
def sel_user_ffs(sdf, user_n):
    sdf_sel_u = sdf[sdf['user'] == user_n]
    ff = sdf_sel_u.groupby("user")
    # print(ff.size())
    return sdf_sel_u

# コマンドラインからどのユーザを選択するか選ぶ
user_n = int(input('\nユーザの選択1~41 >> '))

# 各multi_flagごとに選択したユーザを抽出する
selu_aa = sel_user_ffs(aa, user_n)
selu_ab = sel_user_ffs(ab, user_n)
selu_ac = sel_user_ffs(ac, user_n)
selu_ad = sel_user_ffs(ad, user_n)

selu_ba = sel_user_ffs(ba, user_n)
selu_bb = sel_user_ffs(bb, user_n)
selu_bc = sel_user_ffs(bc, user_n)
selu_bd = sel_user_ffs(bd, user_n)

selu_ca = sel_user_ffs(ca, user_n)
selu_cb = sel_user_ffs(cb, user_n)
selu_cc = sel_user_ffs(cc, user_n)
selu_cd = sel_user_ffs(cd, user_n)

selu_da = sel_user_ffs(da, user_n)
selu_db = sel_user_ffs(db, user_n)
selu_dc = sel_user_ffs(dc, user_n)
selu_dd = sel_user_ffs(dd, user_n)

'''
さてどうしようか
'''