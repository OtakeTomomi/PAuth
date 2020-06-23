基本的にSTEP1〜STEP4を一度ずつ実行すれば実験用ファイルはできる...はず
実験用データの欠損値確認とかを別途行う必要がある(2020/0427)

(venv) (base) tomomi-mbp:10_feature_selection otaketomomi$ python 04_add_flag.py
欠損値削除前→行数：21117 列数：106
100%|████████████████████████████████████████████████████| 21117/21117 [01:18<00:00, 269.77it/s]
21117
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 21117 entries, 0 to 21116
Columns: 107 entries, user to multi_flag
dtypes: float64(98), int64(9)
memory usage: 17.2 MB
None
欠損値削除後→行数：20580 列数：107
(venv) (base) tomomi-mbp:10_feature_selection otaketomomi$



<documentごとに切り分けたもの>


(venv) (base) tomomi-mbp:10_feature_selection otaketomomi$ python 04_add_flag.py 
欠損値削除前→行数：20926 列数：106
100%|████████████████████████████████████| 20926/20926 [01:19<00:00, 262.11it/s]
20926
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20926 entries, 0 to 20925
Columns: 107 entries, user to multi_flag
dtypes: float64(98), int64(9)
memory usage: 17.1 MB
None
欠損値削除後→行数：20399 列数：107
(venv) (base) tomomi-mbp:10_feature_selection otaketomomi$ 





(venv) (base) tomomi-mbp:10_feature_selection otaketomomi$ python 01.5_document_split.py 
1:[1, 2, 3, 4, 5]
2:[1, 2, 3, 4, 5, 6, 7]
3:[1, 2, 3, 4, 5, 6, 7]
4:[1, 2, 3, 4, 5]
5:[1, 2, 3, 4, 5, 6, 7]
6:[1, 2, 3, 4, 5, 6, 7]
7:[1, 2, 3, 4, 5]
8:[1, 2, 3, 4, 5, 6, 7]
9:[1, 2, 3, 4, 5, 6, 7]
10:[1, 2, 3, 4, 5, 6, 7]
11:[1, 2, 3, 4, 5, 6, 7]
12:[1, 2, 3, 4, 5]
13:[1, 2, 3, 4, 5]
14:[1, 2, 3, 4, 5]
15:[1, 2, 3, 4, 5]
16:[1, 2, 3, 4, 5, 6, 7]
17:[1, 2, 3, 4, 5]
18:[1, 2, 3, 4, 5]
19:[1, 2, 3, 4, 5]
20:[1, 2, 3, 4, 5]
21:[1, 2, 3, 4, 5, 6, 7]
22:[1, 2, 3, 4, 5]
23:[1, 2, 3, 4, 5]
24:[1, 2, 3, 4, 5]
25:[1, 2, 3, 4, 5]
26:[1, 2, 3, 4, 5]
27:[1, 2, 3, 4, 5, 6, 7]
28:[1, 2, 3, 4, 5, 6, 7]
29:[1, 2, 3, 4, 5]
30:[1, 2, 3, 4, 5]
31:[1, 2, 3, 4, 5]
32:[1, 2, 3, 4, 5, 6, 7]
33:[1, 2, 3, 4, 5]
34:[1, 2, 3, 4]
35:[1, 2, 3, 4, 5, 6, 7]
36:[1, 2, 3, 4, 5]
37:[1, 2, 3, 4, 5]
38:[1, 2, 3, 4, 5]
39:[1, 2, 3, 4, 5]
40:[1, 2, 3, 4, 5]
41:[1, 2, 3, 4, 5]
OK
(venv) (base) tomomi-mbp:10_feature_selection otaketomomi$


# いったんボツ
