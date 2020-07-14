import numpy as np
import pandas as pd
from pandas import DataFrame

# sd == 各multi_flagのdata
def conf_sel_flag_qty(sd):
    user = sd.groupby("user")
    index = DataFrame(user.size().sort_values(ascending = False))
    list_index = index.index.values
    print('選択したflagに含まれるメンバー(データの多い順)\n',list_index)
    return list_index

def conf_data(sdf, sdf_f, sdf_f_sel, ls_sdf_f):
    # 実験に使用するデータの確認
    print('all:', sdf['user'].count())
    print('all - select_data:', sdf_f['user'].count())
    print('select_data:', sdf['user'].count() - sdf_f['user'].count())
    print('select_hazure_data:', sdf_f_sel['user'].count())
    print('\n外れ値として扱うuser\n',ls_sdf_f)

def conf_matome(X, Y, X_train, Y_train, X_test, Y_test, X_test_t, X_test_f, Y_test_t, Y_test_f, train_target, test_target):
    print('\nX_base:', X.shape)
    print('Y_base:', Y.shape)
    print('X_train:', X_train.shape)
    print('Y_train:', Y_train.shape)
    print('X_test:', X_test.shape)
    print('Y_test:', Y_test.shape)
    print('true_X_test:', X_test_t.shape)
    print('false_X_test:', X_test_f.shape)
    print('true_Y_test:', Y_test_t.shape)
    print('false_Y_test:', Y_test_f.shape)
    print('y_train:', train_target)
    print('y_test:', test_target)

def conf_outlier(st, st_f, st_f_us):
    print('all:', st['user'].count())
    print('all - select_data:', st_f['user'].count())
    print('select_data:', st['user'].count() - st_f['user'].count())
    print('select_hazure_data:', st_f_us['user'].count())


if __name__ == "__main__":

    # 各multi_flagに含まれる各ユーザのデータ数について確認したい場合
    list_index11= conf_sel_flag_qty(aa)
    list_index12= conf_sel_flag_qty(ab)
    list_index13= conf_sel_flag_qty(ac)
    list_index14= conf_sel_flag_qty(ad)
    list_index21= conf_sel_flag_qty(ba)
    list_index22= conf_sel_flag_qty(bb)
    list_index23= conf_sel_flag_qty(bc)
    list_index24= conf_sel_flag_qty(bd)
    list_index31= conf_sel_flag_qty(ca)
    list_index32= conf_sel_flag_qty(cb)
    list_index33= conf_sel_flag_qty(cc)
    list_index34= conf_sel_flag_qty(cd)
    list_index41= conf_sel_flag_qty(da)
    list_index42= conf_sel_flag_qty(db)
    list_index43= conf_sel_flag_qty(dc)
    list_index44= conf_sel_flag_qty(dd)


    # 実験に使用するデータの確認
    conf_data(sdf, sdf_f, sdf_f_sel, ls_sdf_f)

    conf_matome(X, Y, X_train, Y_train, X_test, Y_test, X_test_t, X_test_f, Y_test_t, Y_test_f, y_train, y_test)
