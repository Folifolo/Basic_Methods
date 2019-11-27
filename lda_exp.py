from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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




from neural_network import load_split

if __name__ == "__main__":
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    train_tp = 0
    train_fp = 0
    train_fn = 0
    train_tn = 0

    diags_norm_rythm = [0]
    diags_fibrilation = [15]
    diags_flutter = [16, 17, 18]
    diags_hypertrophy = [119, 120]
    diags_extrasystole = [69, 70, 71, 72, 73, 74, 75, 86]

    dgs = [diags_norm_rythm, diags_fibrilation, diags_flutter, diags_hypertrophy, diags_extrasystole]
    num_components = 100

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_split([140], rnd_state=42, is_new=False)
    X_train = np.concatenate((X_train, X_val), 0)
    Y_train = np.concatenate((Y_train, Y_val), 0)

    pca = PCA(n_components=X_train.shape[0])
    b = pca.fit_transform(X_train[:,:,0])
    btst = pca.transform(X_test[:,:,0])

    lda = LinearDiscriminantAnalysis()
    lda.fit(b , Y_train[:,0])
    m = find_optimal_param(lda, b, Y_train[:,0])


    tp, fp, fn, tn = predict(lda, b, Y_train[:,0], m)
    test_tp += tp
    test_fp += fp
    test_fn += fn
    test_tn += tn

    tp, fp, fn, tn = predict(lda, btst, Y_test[:,0], m)
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


