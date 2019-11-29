from abc import ABC

from fisher_discriminant import FisherDiscriminantAnalisys
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from utils import *


class MethodUtils(ABC):

    def __init__(self):
        self.method
        #self.m

    def fit(self, x, y):
        self.method.fit(x, y)
        #self.find_optimal_param(x, y)

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

        return probs_train[mx]

    def predict(self, x, m):
        if len(x) != 0:
            probs = self.method.predict_proba(x)[:, 1]

            return probs > m

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


if __name__ == "__main__":
    from dataset import NEW_DIAGNOSIS, load_new_dataset_6002, diagnosis_to_binary

    num_components = 100

    xy = load_new_dataset_6002()
    X = xy["x"]
    Y = xy["y"]

    pca = PCA(n_components=X.shape[0])
    b = pca.fit_transform(X)

    Y_new = np.zeros((Y.shape[0], len(NEW_DIAGNOSIS)))
    for j in range(len(NEW_DIAGNOSIS)):
        Y_new[:, j] = diagnosis_to_binary(Y, NEW_DIAGNOSIS[j])[:, 1]

    Y = Y_new

    for d in range(len(NEW_DIAGNOSIS)):
        test_tp = 0
        test_fp = 0
        test_fn = 0
        test_tn = 0
        train_tp = 0
        train_fp = 0
        train_fn = 0
        train_tn = 0

        for train_index, test_index in cross_val(b.shape[0], 100):
            lda = LdaUtils()
            lda.fit(b[train_index, :num_components], Y[train_index, d])
            m = lda.find_optimal_param(b[train_index, :num_components], Y[train_index, d])

            y_prediction = lda.predict(b[test_index, :num_components], m)
            y_labels = Y[test_index, d]
            tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()
            test_tp += tp
            test_fp += fp
            test_fn += fn
            test_tn += tn

            y_prediction = lda.predict(b[train_index, :num_components], m)
            y_labels = Y[train_index, d]
            tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()
            train_tp += tp
            train_fp += fp
            train_fn += fn
            train_tn += tn

        train_se = train_tp / (train_tp + train_fn)
        train_sp = train_tn / (train_tn + train_fp)
        print("Train. Se = %s, Train. Sp = %s" % (round(train_sp, 4), round(train_se, 4)))
        test_se = test_tp / (test_tp + test_fn)
        test_sp = test_tn / (test_tn + test_fp)
        print("Val. Se = %s, Val. Sp = %s" % (round(test_sp, 4), round(test_se, 4)))
