
import pandas as pd
import numpy as np
pd.options.display.notebook_repr_html = False

# indexとColumnsをリストで指定
data = np.random.randint(0,100,(6, 2))
index = [['a', 'a', 'b', 'b', 'c', 'c'], [1, 2, 1, 2, 1, 2]]
columns = ['data1', 'data2']
df = pd.DataFrame(data,
                  index=index,
                  columns=columns)
print(df)

# indexを２次元の配列とすればマルチインデックスをもつDataFrameを作成可能F
df.index.names = ['Str', 'Int']
print(df)
