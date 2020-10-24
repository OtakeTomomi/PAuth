"""
This program is main program for one class experiment.
combination = True
"""

import pandas as pd

# 内部ライブラリ
from expmodule.dataset import load_frank

# warning ignore code
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
frank_df = load_frank(combination=True)

# データのColumn取得
df_column = frank_df.columns.values

# データをmulti_flagを基準に分割
# a,b,c,dのストローク方向はflag_splitに記載



