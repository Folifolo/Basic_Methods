from dataset import load_dataset, num_to_diag, diag
import os
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt
from  scipy.signal import medfilt
from matplotlib.pyplot import figure
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils import *
from preprocessing import *
def find_optimal_param(lda, x_train, y_train):

    probs_train = lda.predict_proba(x_train)[:, 1]

    y_train = [x for _,x in sorted(zip(probs_train,y_train))]
    y_train = np.array(y_train)
    probs_train.sort()
    Se = []
    Sp = []
    for p in range(len(probs_train)):
        tp = np.count_nonzero(y_train[p:] == 1)
        fp = np.count_nonzero(y_train[p:] == 0)
        tn = np.count_nonzero(y_train[:p] == 0)
        fn = np.count_nonzero(y_train[:p] == 1)
        Se.append(tp/(tp+fn))
        Sp.append(tn/(tn+fp))

    mx = np.argmax(-(1-np.array(Sp) - np.array(Se)))

    return probs_train[mx]

def predict(lda, x, y, m):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if len(x) != 0:
        probs= lda.predict_proba(x)[:, 1]

        for j in range(len(x)):
            if probs[j] > m:
                if y[j] == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if y[j] == 1:
                    fn +=1
                else:
                    tn +=1
    """
    if tp+fn == 0:
        Se = 1
    else:
        Se = tp/(tp+fn)
    if tn+fp ==0:
        Sp = 1
    else:
        Sp=tn/(tn+fp)
    """

    return tp, fp, fn, tn

class Branch(object):

    def __init__(self, lda, m, x, y):
        """Constructor"""
        self.lda = lda
        self.m = m
        self.x = x
        self.y = y
        self.left = None
        self.right = None

    def grow(self):
        probs_train = self.lda.predict_proba(self.x)[:, 1]
        y_train = [x for _,x in sorted(zip(probs_train,self.y))]
        y_train = np.array(y_train)
        probs_train.sort()

        middle = (np.abs(probs_train - m)).argmin()

        right_x = self.x[middle:]
        right_y = y_train[middle:]
        left_x = self.x[:middle]
        left_y = y_train[:middle]

        lda_right = LinearDiscriminantAnalysis()
        lda_right.fit(right_x,right_y)

        m_right = find_optimal_param(lda_right, right_x, right_y)


        lda_left = LinearDiscriminantAnalysis()
        lda_left.fit(left_x,left_y)

        m_left = find_optimal_param(lda_left, left_x, left_y)

        right_branch = Branch(lda_right, m_right, right_x, right_y)
        left_branch = Branch(lda_left, m_left, left_x, left_y)

        self.left = left_branch
        self.right = right_branch


    def divide_data(self, x, y):
        probs_train = self.lda.predict_proba(x)[:, 1]
        y_train = [x for _,x in sorted(zip(probs_train,y))]
        y_train = np.array(y_train)
        probs_train.sort()

        middle = (np.abs(probs_train - self.m)).argmin()

        right_x = x[middle:]
        right_y = y_train[middle:]
        left_x = x[:middle]
        left_y = y_train[:middle]

        return [left_x, left_y], [right_x, right_y]


    def predict(self, x, y):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        if self.left == None:
            if len(x) != 0:
                probs= self.lda.predict_proba(x)[:, 1]

                for j in range(len(x)):
                    if probs[j] > m:
                        if y[j] == 1:
                            tp+=1
                        else:
                            fp+=1
                    else:
                        if y[j] == 1:
                            fn +=1
                        else:
                            tn +=1
        else:
            left_data, right_data = self.divide_data(x, y)
            tp_l, fp_l, fn_l, tn_l = self.left.predict(left_data[0], left_data[1])
            tp_r, fp_r, fn_r, tn_r =self.right.predict(right_data[0], right_data[1])
            tp = tp_l + tp_r
            fp = fp_l + fp_r
            fn = fn_l + fn_r
            tn = tn_l + tn_r
        return tp, fp, fn, tn



if __name__ == "__main__":
    num_components = 14

    pkl1 = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\xy_6002.pkl"
    pkl2 = "C:\\Users\\donte_000\\Downloads\\Telegram Desktop\\xy_102.pkl"

    infile = open(pkl1, 'rb')
    xy = pkl.load(infile)
    infile.close()

    X_new = xy["x"]
    Y = xy["y"]
    X = np.zeros((X_new.shape[0], 14))
    X[:,:1] = X_new[:,:1]
    X[:,1:3] = X_new[:,249:251]
    X[:,3:5] = X_new[:,499:501]
    X[:,5:7] = X_new[:,749:751]
    X[:,7:9] = X_new[:,999:1001]
    X[:,9:11] = X_new[:,1249:1251]
    X[:,11:12] = X_new[:,1499:1500]
    X[:,12:14] = X_new[:,-2:]
    #pca = PCA(n_components=X.shape[1])
    #b = pca.fit_transform(X)
    b = X
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
            lda = LinearDiscriminantAnalysis()
            lda.fit(b[train_index, :num_components],Y[train_index,d])
            m = find_optimal_param(lda, b[train_index, :num_components], Y[train_index,d])

            tree = Branch(lda, m, b[train_index, :num_components], Y[train_index,d])
            #tree.grow()
            tp, fp, fn, tn = tree.predict(b[test_index, :num_components], Y[test_index,d])
            test_tp += tp
            test_fp += fp
            test_fn += fn
            test_tn += tn

            tp, fp, fn, tn = tree.predict(b[train_index, :num_components], Y[train_index,d])
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
