"""
学習データとテストデータに分けるときに必要となる

session→user→data数で絞り込みをかけていく
"""


# これは共通
def session(df_flag, select_session='all'):
    df_session = df_flag.copy()
    # docが1~5のもの
    if select_session == 'first':
        df_first = df_session.query('1<= doc < 6')
        return df_first
    # docが6or7のもの
    elif select_session == 'latter':
        df_latter = df_session.query('6 <= doc < 8')
        return df_latter
    # 全部
    elif select_session == 'all':
        df_all = df_session
        return df_all


if __name__ == '__main__':
    print("This is module! The name is session_select")
