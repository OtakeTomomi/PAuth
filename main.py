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

