import numpy as np


class FisherDiscriminantAnalisys(object):

    def __init__(self):
        """Constructor"""

    def fit(self, X, y):
        means = []
        cov = []
        self.priors_ = np.bincount(y.astype(int)) / float(len(y))
        for ind in [0, 1]:
            Xg = X[y == ind, :]
            meang = Xg.mean(axis=0)
            means.append(meang)

            if len(Xg) == 1:
                cov1 = np.cov(np.concatenate((Xg.T, Xg.T), 1))
            else:
                cov1 = np.cov(Xg.T)
            cov.append(cov1)

        self.means_ = np.asarray(means)
        self.covariations_ = np.asarray(cov)
        self.w = np.dot(np.linalg.inv(self.covariations_[0] + self.covariations_[1]), (self.means_[1] - self.means_[0]))

    def _predict(self, X):

        X = np.array(X)
        res = np.dot(X, self.w)
        return res

    def predict(self, X, c):
        return (self._predict(X) > c).astype(int)

    def predict_proba(self, X):
        values = self._predict(X)
        values = np.expand_dims(values, 1)
        values = np.concatenate((-values, values), 1)
        return values


if __name__ == "__main__":
    X = np.array([[-1, -1], [-2, -1], [-3, -4], [1, 1], [2, 1], [3, 4]])

    y = np.array([0, 0, 0, 1, 1, 1])
    clf = FisherDiscriminantAnalisys()
    clf.fit(X, y)
    print(clf.predict([[-1, 1], [1, 3], [5, -1], [2, 0.8], [3, 0.8]], 0))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    print(clf.predict([[-1, 1], [1, 3], [5, -1], [2, 0.8], [3, 0.8]]))
