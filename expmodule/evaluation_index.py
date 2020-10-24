import numpy as np


def far_frr_ber(normal_result, anomaly_result):
    tp = np.count_nonzero(normal_result == 1)
    fn = np.count_nonzero(normal_result == -1)
    fp = np.count_nonzero(anomaly_result == 1)
    tn = np.count_nonzero(anomaly_result == -1)
    re_far = fp / (tn + fp)
    re_frr = fn / (fn + tp)
    re_ber = 0.5 * ((fp / (tn + fp)) + (fn / (fn + tp)))

    # accuracy = ((TP+TN)/(TP+FN+FP+TN))
    # print(accuracy)
    return re_far, re_frr, re_ber


if __name__ == "__main__":
    print("This is module! The name is evaluation_index")
