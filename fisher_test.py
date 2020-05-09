from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from preprocessing import *
from utils import *
from sklearn.metrics import confusion_matrix, roc_curve


def find_optimal_param(fda, x_train, y_train):
    probs_train = fda.predict_proba(x_train)[:,1]

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


def predict(fda, x, m):
    if len(x) != 0:
        probs = fda.predict_proba(x)[:,1]

        pr_true = (probs > m).astype(int)

    return pr_true


if __name__ == "__main__":
    from dataset import load_dataset, load_new_dataset_6002, diagnosis_to_binary, MOST_FREQ_DIAGS_NUMS
    from fisher_discriminant import FisherDiscriminantAnalisys
    num_components = 100

    xy = load_dataset()

    Y = xy["y"]
    outfile = open('C:\\Users\\donte_000\\PycharmProjects\\ClassificationECG\\data\\6002_norm_old_old.pkl', 'rb')
    X = pkl.load(outfile)
    outfile.close()
    mn = X.mean(axis = 0)
    st = X.std(axis = 0)
    x_std = np.zeros(X.shape)
    for i in range(st.shape[0]):
        if st[i] == 0:
            st[i] = 1
    for i in range(X.shape[0]):
        x_std[i] = (X[i] - mn)/st

    X = x_std
    pca = PCA(n_components=X.shape[0])
    b = pca.fit_transform(X)
    '''
    fda = Fd()
    fda.fit(b[:, :num_components], Y[:, 140])
    m = find_optimal_param(fda, b[:, :num_components], Y[:, 140])

    from dataset import NEW_DIAGNOSIS, load_new_dataset_6002, diagnosis_to_binary
    import matplotlib.pyplot as plt
    num_components = 100
    print(Y[:, 140].sum())
    xy = load_new_dataset_6002()
    X1 = xy["x"]
    Y1 = xy["y"]

    b = pca.transform(X1)

    y_prediction = predict(fda, b[:, :num_components], m)
    y_labels = Y1[:,15]
    print(Y1[:, 15].sum())
    tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()


    test_se = tp / (tp + fn)
    test_sp = tn / (tn + fp)
    print("Val. Se = %s, Val. Sp = %s" % (round(test_sp, 4), round(test_se, 4)))


    y_prediction = fda.predict_proba(b[:, :num_components])
    fpr, tpr, thresholds = roc_curve(y_labels, y_prediction)
    plt.plot(fpr, tpr)

    plt.scatter(fp/(fp+tn), test_se)
    plt.show()
'''

    for d in reversed(MOST_FREQ_DIAGS_NUMS):
        y_prediction =[]
        y_labels = []
        for train_index, test_index in cross_val(b.shape[0], 1):
            fda = FisherDiscriminantAnalisys()
            fda.fit(b[train_index, :num_components], Y[train_index, d])
            m = find_optimal_param(fda, b[train_index, :num_components], Y[train_index, d])

            y_prediction.append(predict(fda, b[test_index, :num_components], m))
            y_labels.append(Y[test_index, d])

        y_prediction = np.array(y_prediction).flatten()
        y_labels = np.array(y_labels).flatten()
        tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()

        test_se = tp / (tp + fn)
        test_sp = tn / (tn + fp)
        print("Val. Se = %s, Val. Sp = %s" % (round(test_sp, 4), round(test_se, 4)))
