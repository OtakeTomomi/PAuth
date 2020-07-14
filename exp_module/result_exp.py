import numpy as np

def far_frr(normal_result, anomaly_result):
    TP = np.count_nonzero(normal_result == 1)
    FN = np.count_nonzero(normal_result == -1)
    FP = np.count_nonzero(anomaly_result == 1)
    TN = np.count_nonzero(anomaly_result == -1)
    FRR = FN / (FN + TP)
    FAR = FP / (TN + FP)
    BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
    return FRR, FAR, BER

if __name__ == "__main__":
    print('result_exp module')