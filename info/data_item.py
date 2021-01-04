import pandas as pd
import numpy as np

#

data_item = pd.read_csv("data_item.csv", sep=",", header=None)
data_item.columns = ['user', 'flag', 'data']
# print(data_item)
# print(data_item.T)
columns = ['user', 'aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'bd', 'ca', 'cb', 'cc', 'cd', 'da', 'db', 'dc', 'dd']

data_item_new = pd.DataFrame(np.zeros((41, 17)))
data_item_new.columns = columns
# print(data_item_new)

data_item_m = data_item.set_index(['user', 'flag'])
# print(data_item_m)

user = pd.DataFrame([i for i in range(1, 42)])
# print(user)

co = pd.DataFrame(columns[1:])
# print(list(data_item_m['data'].xs([1], level=['user'])))

for i in range(1, 42):
    data = pd.DataFrame(list(data_item_m['data'].xs([i], level=['user'])))
    # print(data.T)
    co[i] = data
    # user = pd.concat([pd.DataFrame([i]), data.T])
# print(co.T)
cot = co.T
# cot['user'] = user
# print(co)
cot.to_csv('data_item_new.csv', header=None)


df = pd.read_csv('data_item_new.csv')
df.columns = columns
# print(df)

# table = {'aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'bd', 'ca', 'cb', 'cc', 'cd', 'da', 'db', 'dc', 'dd'}
# for k in range(1, 42):
#     for i in range(16):
#         table[columns[i+1]] = data_item[2] == [data_item[data_item[0] == k] & data_item[data_item[1] == columns[i+1]]]
# print(table)

# df3 = {'aa', 'ab', 'ac', 'ad', 'ba', 'bb', 'bc', 'bd', 'ca', 'cb', 'cc', 'cd', 'da', 'db', 'dc', 'dd'}

heikin = []
min = []
max = []
data_std = []
data_count = []
user_data = []

for i, item in enumerate(columns[1:]):
    df2 = df[df[item] >= 0]
    heikin.append(df2[item].mean())
    min.append(df2[item].min())
    max.append(df2[item].max())
    data_std.append(df2[item].std())
    data_count.append(df2[item].count())
    user_data.append(list(df2['user']))

    # print(df2[item])
    # print(item)
print(heikin)
print(min)
print(max)
print(data_std)
print(user_data)

table = pd.concat([pd.DataFrame(columns[1:]), pd.DataFrame(heikin), pd.DataFrame(min), pd.DataFrame(max),
                   pd.DataFrame(data_std), pd.DataFrame(data_count)], axis=1)
table.columns = ['flag', 'mean', 'min', 'max', 'std', 'user数']
print(table)

table.to_csv('table.csv')
