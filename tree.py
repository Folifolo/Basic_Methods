from sklearn.decomposition import PCA

from preprocessing import *
from utils import *
from abc import ABC, abstractmethod
from methodutils import FdaUtils, LdaUtils


class TreeNode(ABC):


    def __init__(self):
        """Constructor"""
        self.method
        self.left_child = None
        self.right_child = None

    def is_have_childs(self):
        return self.left_child is not None and self.right_child is not None

    def grow(self):
        """Set left and right"""
        # self.right_child = TreeNode()
        # self.left_child = TreeNode()

    def fit(self, x, y):
        self.method.fit(x, y)

        if self.is_have_childs():
            left, right = self.divide_data(x)
            if max(y[left]) == 0 or min(y[right]) == 1:
                self.left_child = self.right_child = None
            else:
                self.right_child.fit(x[left], y[left])
                self.left_child.fit(x[right], y[right])

    def divide_data(self, x):
        probs = self.method.predict_proba(x)[:, 1]
        left = (probs <= self.method.m)
        right = (probs > self.method.m)
        return left, right

    def predict(self, x):
        if not self.is_have_childs():
            pred = self.method.predict(x)

        elif self.is_have_childs():
            left, right = self.divide_data(x)
            l_pred = self.left_child.predict(x[left])
            r_pred = self.right_child.predict(x[right])
            pred = np.ones(x.shape[0]) * 2
            pred[left] = l_pred
            pred[right] = r_pred

        return pred


class FdaTree(TreeNode):
    def __init__(self):
        """Constructor"""
        self.method = FdaUtils()
        self.left_child = None
        self.right_child = None

    def grow(self):
        self.right_child = FdaTree()
        self.left_child = FdaTree()


class LdaTree(TreeNode):
    def __init__(self):
        """Constructor"""
        self.method = LdaUtils()
        self.left_child = None
        self.right_child = None

    def grow(self):
        self.right_child = LdaTree()
        self.left_child = LdaTree()


if __name__ == "__main__":
    np.seterr(all='raise')
    from sklearn.metrics import confusion_matrix
    from dataset import load_dataset, MOST_FREQ_DIAGS_NUMS_NEW

    num_components = 100

    infile = open('C:\\Users\\donte_000\\PycharmProjects\\Basic_Methods\\data\\data_old_and_new_without_noise.pkl',
                  'rb')
    (old, new) = pkl.load(infile)
    infile.close()

    Y = old["y"]
    outfile = open('C:\\Users\\donte_000\\PycharmProjects\\Basic_Methods\\data\\6002_old_Dif.pkl', 'rb')
    X = pkl.load(outfile)
    outfile.close()
    pca = PCA(n_components=X.shape[0])
    b = pca.fit_transform(X)

    for d in reversed(MOST_FREQ_DIAGS_NUMS_NEW):
        y_prediction = []
        y_labels = []
        for train_index, test_index in cross_val(b.shape[0], 500):
            tree = FdaTree()
            tree.grow()
            tree.fit(b[train_index, :num_components], Y[train_index, d])

            y_prediction.append(tree.predict(b[test_index, :num_components]))
            y_labels.append(Y[test_index, d])

        y_prediction = np.array(y_prediction).flatten()
        y_labels = np.array(y_labels).flatten()
        tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()

        test_se = tp / (tp + fn)
        test_sp = tn / (tn + fp)
        print("Val. Se = %s, Val. Sp = %s" % (round(test_sp, 4), round(test_se, 4)))
