from abc import ABC

from fisher_discriminant import FisherDiscriminantAnalisys
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from utils import *


class MethodUtils(ABC):

    def __init__(self):
        self.method
        self.m

    def fit(self, x, y):
        self.method.fit(x, y)
        self.find_optimal_param(x, y)

    def find_optimal_param(self, x_train, y_train):
        probs_train = self.method.predict_proba(x_train)[:, 1]

        y_train = [x for _, x in sorted(zip(probs_train, y_train))]
        y_train = np.array(y_train)
        probs_train.sort()
        Se = []
        Sp = []
        for p in range(len(probs_train)):
            tp = np.count_nonzero(y_train[p:] == 1)
            fp = np.count_nonzero(y_train[p:] == 0)
            tn = np.count_nonzero(y_train[:p] == 0)
            fn = np.count_nonzero(y_train[:p] == 1)
            Se.append(tp / (tp + fn))
            Sp.append(tn / (tn + fp))

        mx = np.argmax(-(1 - np.array(Sp) - np.array(Se)))

        self.m = probs_train[mx]

    def predict(self, x):
        if len(x) != 0:
            probs = self.method.predict_proba(x)[:, 1]

            return (probs > self.m).astype(int)

        else:
            return 0

    def predict_proba(self, x):
        return self.method.predict_proba(x)


class LdaUtils(MethodUtils):
    def __init__(self):
        self.method = LinearDiscriminantAnalysis()


class FdaUtils(MethodUtils):
    def __init__(self):
        self.method = FisherDiscriminantAnalisys()

