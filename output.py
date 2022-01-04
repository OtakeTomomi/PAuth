"""
図の作成
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
# import seaborn as sns
# from matplotlib import gridspec


def main(PATH, plotfile, n):
    # 結果の読み込み
    re_mean_val = pd.read_csv(f'{PATH}result_mean_val.csv', sep=",", header=None)
    re_max_val = pd.read_csv(f'{PATH}result_max_val.csv', sep=",", header=None)
    re_min_val = pd.read_csv(f'{PATH}result_min_val.csv', sep=",", header=None)
    re_std_val = pd.read_csv(f'{PATH}result_std_val.csv', sep=",", header=None)

    re_mean_test = pd.read_csv(f'{PATH}result_mean_test.csv', sep=",", header=None)
    re_max_test = pd.read_csv(f'{PATH}result_max_test.csv', sep=",", header=None)
    re_min_test = pd.read_csv(f'{PATH}result_min_test.csv', sep=",", header=None)
    re_std_test = pd.read_csv(f'{PATH}result_std_test.csv', sep=",", header=None)

    # columnsの内訳
    columns_val = ['scenario', 'flag', 'performance', 'model', 'Accuracy', 'Precision',
                   'Recall', 'F1', 'AUC', 'FAR', 'FRR', 'BER']

    columns_test = ['scenario', 'flag', 'performance', 'model', 'AUC', 'Accuracy', 'BER',
                    'F1', 'FAR', 'FRR', 'Precision', 'Recall']

    model_index = ['LocalOutlierFactor', 'IsolationForest', 'OneClassSVM', 'EllipticEnvelope']

    # columnsの設定
    re_mean_val.columns = columns_val
    re_max_val.columns = columns_val
    re_min_val.columns = columns_val
    re_std_val.columns = columns_val

    re_mean_test.columns = columns_test
    re_max_test.columns = columns_test
    re_min_test.columns = columns_test
    re_std_test.columns = columns_test

    # performanceの削除
    # re_mean_val = re_mean_val.drop('performance', axis=1)
    # re_max_val = re_max_val.drop('performance', axis=1)
    # re_min_val = re_min_val.drop('performance', axis=1)
    # re_std_val = re_std_val.drop('performance', axis=1)
    #
    # re_mean_test = re_mean_test.drop('performance', axis=1)
    # re_max_test = re_max_test.drop('performance', axis=1)
    # re_min_test = re_min_test.drop('performance', axis=1)
    # re_std_test = re_std_test.drop('performance', axis=1)

    # multi_indexの設定
    re_mean_val = re_mean_val.set_index(['scenario', 'flag', 'performance', 'model'])
    re_max_val = re_max_val.set_index(['scenario', 'flag', 'performance', 'model'])
    re_min_val = re_min_val.set_index(['scenario', 'flag', 'performance', 'model'])
    re_std_val = re_std_val.set_index(['scenario', 'flag', 'performance', 'model'])

    re_mean_test = re_mean_test.set_index(['scenario', 'flag', 'performance', 'model'])
    re_max_test = re_max_test.set_index(['scenario', 'flag', 'performance', 'model'])
    re_min_test = re_min_test.set_index(['scenario', 'flag', 'performance', 'model'])
    re_std_test = re_std_test.set_index(['scenario', 'flag', 'performance', 'model'])

    os.makedirs(f'{PATH}{plotfile}', exist_ok=True)

    def plot_re(re_mean_m, re_max_m, re_min_m, re_std_m, p):
        # multi_flag[11,13,22,24,31,33,42,44]
        # これはあとで一括管理
        scenario_list = ['first', 'latter', 'all', 'all_test_shinario2']
        sessions = {'first': 'intra', 'latter': 'inter', 'all': 'combined', 'all_test_shinario2': 'combined2'}
        # s = 'first'
        # k = 3
        # m = ['_', 'up', 'right', 'down', 'left']
        m = ['-', 'up', 'left', 'down', 'right', 'all_flag']
        # plt.figure(1)

        for s in scenario_list:
            # plt.subplot(4, 1, 1)
            if n == 4:
                flag_list1 = [0, 1, 2, 3, 4, 5]
                flag_list2 = list(re_mean_m.index.get_level_values('flag'))
                flag_list = sorted(list(set(flag_list1) & set(flag_list2)))
            else:
                flag_list1 = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44, 55]
                flag_list2 = list(re_mean_m.index.get_level_values('flag'))
                flag_list = sorted(list(set(flag_list1) & set(flag_list2)))
            for k in flag_list:

                plt.figure(figsize=(6*1, 6*2))
                # gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
                plt.subplot(211)
                # 図のプロット
                # x軸のラベル
                labels = list(re_mean_m.index.get_level_values('model'))[:4]

                # y軸_mean
                y_mean_ber = list(re_mean_m['BER'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                # print(y_mean_ber)
                # print(y_mean_ber)
                y_mean_far = list(re_mean_m['FAR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_mean_frr = list(re_mean_m['FRR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_mean_acc = list(re_mean_m['Accuracy'].xs([s, k, p], level=['scenario', 'flag', 'performance']))

                # y軸_max
                y_max_ber = list(re_max_m['BER'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_max_far = list(re_max_m['FAR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_max_frr = list(re_max_m['FRR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))

                # y軸_mean
                y_min_ber = list(re_min_m['BER'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_min_far = list(re_min_m['FAR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_min_frr = list(re_min_m['FRR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))

                # y軸_std
                y_std_ber = list(re_std_m['BER'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_std_far = list(re_std_m['FAR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_std_frr = list(re_std_m['FRR'].xs([s, k, p], level=['scenario', 'flag', 'performance']))
                y_std_acc = list(re_std_m['Accuracy'].xs([s, k, p], level=['scenario', 'flag', 'performance']))

                # タイトル
                if n == 4:
                    plt.title(f'scenario: {sessions[s]}, flag: {m[k]}')
                else:
                    plt.title(f'scenario: {sessions[s]}, flag: {m[k//10]}+{m[k%10]}')

                x = np.arange(len(labels))
                width = 0.2

                error_bar_set = dict(lw=1, capthick=1, capsize=2)

                # 棒グラフ(mean)
                plt.bar(x, y_mean_ber, width=width, align='center', label='BER', alpha=0.7,
                        yerr=y_std_ber, error_kw=error_bar_set, color="white", edgecolor='black')
                plt.bar(x+width, y_mean_far, width=width, align='center', label='FAR', alpha=0.7, hatch="///",
                        yerr=y_std_far, error_kw=error_bar_set, color="white", edgecolor='black')
                plt.bar(x+width*2, y_mean_frr, width=width, align='center', label='FRR', alpha=0.7, hatch="-"*3,
                        yerr=y_std_frr, error_kw=error_bar_set, color="white", edgecolor='black')

                # # 点(max)
                # plt.scatter(x, y_max_ber, marker="$-$")
                # plt.scatter(x+width, y_max_far, marker="$-$")
                # plt.scatter(x+width*2, y_max_frr, marker="$-$")
                #
                # # 点(min)
                # plt.scatter(x, y_min_ber, marker="$-$")
                # plt.scatter(x+width, y_min_far, marker="$-$")
                # plt.scatter(x+width*2, y_min_frr, marker="$-$")
                #
                # plt.vlines(x, ymin=y_min_ber, ymax=y_max_ber, alpha=0.5)
                # plt.vlines(x+width, ymin=y_min_far, ymax=y_max_far, alpha=0.5)
                # plt.vlines(x+width*2, ymin=y_min_frr, ymax=y_max_frr, alpha=0.5)

                # x軸のメモリをラベルで置換
                plt.xticks(x + width/2, labels)

                # y軸のメモリ
                plt.yticks(np.arange(0.0, 1.1, 0.1))

                # plt.ylim(0.0, 1.0)

                # 凡例
                plt.legend(loc='best')

                mean = np.array([y_mean_acc, y_mean_ber, y_mean_far, y_mean_frr])
                # print(mean)
                mean = np.round(mean, 3)
                # mean.columns = labels
                plt.subplot(212)
                plt.axis('off')
                plt.table(cellText=mean,
                          colWidths=[0.225] * 4,
                          rowLabels=['Accuracy', 'BER', 'FAR', 'FRR'],
                          colLabels=['LOF', 'IF', 'OCSVM', 'EE'],
                          loc='upper right')
                # fig.tight_layout()
                # 保存
                if n == 4:
                    name = 'OneclassOne'
                else:
                    name = 'OneclassTwo'
                plt.savefig(f'{PATH}{plotfile}/{name}_{p}_{sessions[s]}_{k}.png')

                # 描画
                # plt.show()
                plt.close()

                img = Image.open(f'{PATH}{plotfile}/{name}_{p}_{sessions[s]}_{k}.png')
                # (left, upper, right, bottom)
                # print(img.size)
                box = (0, 100, 600, 770)
                new_img = img.crop(box)
                # 画像表示
                # new_img.show()
                new_img.save(f'{PATH}{plotfile}/{name}_{p}_{sessions[s]}_{k}.png')

    plot_re(re_mean_val, re_max_val, re_min_val, re_std_val, p='val')
    plot_re(re_mean_test, re_max_test, re_min_test, re_std_test, p='test')




def main2(PATH, plotfile, n):
    # 結果の読み込み
    re_mean_val = pd.read_csv(f'{PATH}result_mean_val.csv', sep=",", header=None)
    re_max_val = pd.read_csv(f'{PATH}result_max_val.csv', sep=",", header=None)
    re_min_val = pd.read_csv(f'{PATH}result_min_val.csv', sep=",", header=None)
    re_std_val = pd.read_csv(f'{PATH}result_std_val.csv', sep=",", header=None)

    re_mean_test = pd.read_csv(f'{PATH}result_mean_test.csv', sep=",", header=None)
    re_max_test = pd.read_csv(f'{PATH}result_max_test.csv', sep=",", header=None)
    re_min_test = pd.read_csv(f'{PATH}result_min_test.csv', sep=",", header=None)
    re_std_test = pd.read_csv(f'{PATH}result_std_test.csv', sep=",", header=None)

if __name__ == '__main__':
    # print("結果")
    PATH = 'result2022/zikken_3-3-all/matome_one/'
    # PATH = 'result2022/zikken_3-4-all/matome_two/'
    plotfile = 'plot_result'
    # combのときはn=16にする
    main(PATH, plotfile, n=4)
    # main(PATH, plotfile, n=16)


