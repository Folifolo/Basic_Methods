import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from preprocessing import *
from utils import *


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

    return tp, fp, fn, tn

from methodutils import FdaUtils

class FDA_node(object):

    def __init__(self):
        """Constructor"""
        self.method = FdaUtils()
        self.left = None
        self.right = None
        self.m = 0.5

    def grow(self):
        self.right = FDA_node()
        self.left = FDA_node()

    def find_optimal_param(self, x, y):
        self.m = self.method.find_optimal_param(x, y)


        if self.left != None and self.right != None:
            left, right = self.divide_data(x)
            self.left.find_optimal_param(x[left], y[left])
            self.right.find_optimal_param(x[right], y[right])


    def fit(self, x, y):
        self.method.fit(x, y)

        if self.left != None and self.right != None:
            left, right = self.divide_data(x)
            if (max(y[left]) == 0 or min(y[right]) == 1):
                self.left = self.right = None
            else:
                self.right.fit(x[left], y[left])
                self.left.fit(x[right], y[right])


    def divide_data(self, x):
        probs = self.method.predict_proba(x)[:, 1]
        left = (probs <= self.m)
        right = (probs > self.m)
        return left, right


    def predict(self, x):
        if self.left == None and self.right == None:
            pred = self.method.predict(x, self.m)

        elif self.left != None and self.right != None:
            left, right = self.divide_data(x)
            l_pred = self.left.predict(x[left])
            r_pred =self.right.predict(x[right])
            pred = np.ones(x.shape[0])*2
            pred[left] = l_pred
            pred[right] = r_pred

        return pred



if __name__ == "__main__":
    np.seterr(all='raise')
    from sklearn.metrics import confusion_matrix
    from dataset import load_dataset, load_new_dataset_6002, diagnosis_to_binary, MOST_FREQ_DIAGS_NUMS_NEW
    from fisher_discriminant import FisherDiscriminantAnalisys
    num_components = 100

    infile = open('C:\\Users\\donte_000\\PycharmProjects\\Basic_Methods\\data\\data_old_and_new_without_noise.pkl', 'rb')
    (old, new) = pkl.load(infile)
    infile.close()

    Y = old["y"]
    outfile = open('C:\\Users\\donte_000\\PycharmProjects\\\ClassificationECG\\data\\6002_old_Dif.pkl', 'rb')
    X = pkl.load(outfile)
    outfile.close()
    pca = PCA(n_components=X.shape[0])
    b = pca.fit_transform(X)



    for d in reversed(MOST_FREQ_DIAGS_NUMS_NEW):
        y_prediction =[]
        y_labels = []
        for train_index, test_index in cross_val(b.shape[0], 1):
            tree = FDA_node()
            tree.grow()
            tree.fit(b[train_index, :num_components],Y[train_index,d])
            tree.find_optimal_param(b[train_index, :num_components], Y[train_index,d])

            y_prediction.append(tree.predict(b[test_index, :num_components]))
            y_labels.append(Y[test_index, d])

        y_prediction = np.array(y_prediction).flatten()
        y_labels = np.array(y_labels).flatten()
        tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()

        test_se = tp / (tp + fn)
        test_sp = tn / (tn + fp)
        print("Val. Se = %s, Val. Sp = %s" % (round(test_sp, 4), round(test_se, 4)))
