import numpy as np
import matplotlib.pyplot as plt


class Lda(object):

    def __init__(self):
        """Constructor"""

    def fit(self, X, y):
        means = []
        cov = []
        self.priors_ = np.bincount(y) / float(len(y))
        for ind in [0, 1]:
            Xg = X[y == ind, :]
            meang = Xg.mean(axis = 0)
            means.append(meang)
            Xc = Xg - meang
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

            S2 = (S ** 2) / (len(Xg) - 1)
            cov.append(np.dot(S2 * Vt.T, Vt))

        self.means_ = np.asarray(means)
        self.covariations_ = np.asarray(cov)

    def _predict(self, X):#, T):
        X = np.array(X)
        Xm0 = X - self.means_[0]
        Xm1 = X - self.means_[1]
        Cov0 = np.linalg.inv(self.covariations_[0])
        Cov1 = np.linalg.inv(self.covariations_[1])
        D = np.dot(np.dot(Xm0, Cov0), Xm0.T)
        res = [0,0]
        res[0] = np.dot(np.dot(self.means_[0], Cov0), X.T)-0.5*np.dot(np.dot(self.means_[0], Cov0), self.means_[0].T) + np.log(self.priors_[0])
        res[1] = np.dot(np.dot(self.means_[1], Cov1), X.T)-0.5*np.dot(np.dot(self.means_[1], Cov0), self.means_[1].T) + np.log(self.priors_[1])
        res = np.array(res)
        return res
    def predict(self, X):
        return self._predict(X).argmax(0)
    def predict_proba(self, X):
        values = self._predict(X)
        values = np.swapaxes(values, 1,0)
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]


X = np.array([[-1, -1], [-2, -1], [-3, -4], [1, 1], [2, 1], [3, 4]])

y = np.array([0, 0, 0, 1, 1, 1])
#print(np.cov(X[y == 1, :].T))
clf = Lda()
clf.fit(X, y)
print(clf.predict([[5, -1], [2, 0.8], [3, 0.8]]))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
print(clf.predict([[-0.8, -1], [2, 0.8]]))
