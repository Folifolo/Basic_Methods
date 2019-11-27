import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from preprocessing import *
from utils import *


def tmp(b,y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if len(b) != 0:
        for j in range(len(b)):
            if b[j] == 1:
                if y[j] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if y[j] == 1:
                    fn +=1
                else:
                    tn +=1
    return tp, fp, fn, tn


if __name__ == "__main__":
    num_components = 100

    pkl1 = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\xy_6002.pkl"
    pkl2 = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\xy_102.pkl"

    infile = open(pkl1, 'rb')
    xy = pkl.load(infile)
    infile.close()

    X = xy["x"]
    Y = xy["y"]
    pca = PCA(n_components=X.shape[0])
    b = pca.fit_transform(X)
    counter = 0
    for i in range(len(Y)):
        counter+=Y[i,14]
    print(counter)
    for d in np.arange(0,15):
        test_tp = 0
        test_fp = 0
        test_fn = 0
        test_tn = 0
        train_tp = 0
        train_fp = 0
        train_fn = 0
        train_tn = 0

        for train_index, test_index in cross_val(b.shape[0], 1):
            log_reg = LogisticRegression()
            log_reg.fit(b[train_index, :num_components],Y[train_index,d])
            log_reg.predict(b[test_index, :num_components])

            tp, fp, fn, tn = tmp(log_reg.predict(b[test_index, :num_components]), Y[test_index,d])
            test_tp += tp
            test_fp += fp
            test_fn += fn
            test_tn += tn

            tp, fp, fn, tn = tmp(log_reg.predict(b[train_index, :num_components]), Y[train_index,d])
            train_tp += tp
            train_fp += fp
            train_fn += fn
            train_tn += tn

        train_se = train_tp/(train_tp+train_fn)
        train_sp=train_tn/(train_tn+train_fp)
        print("Train. Se = %s, Train. Sp = %s" %(round(train_sp,4),  round(train_se,4)))
        test_se = test_tp/(test_tp+test_fn)
        test_sp=test_tn/(test_tn+test_fp)
        print("Val. Se = %s, Val. Sp = %s" %(round(test_sp, 4),  round(test_se, 4)))