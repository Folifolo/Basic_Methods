from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from dataset import load_split, DATA_SIZE


def build_model1(size):
    input_ecg = Input(shape=(size, 1))

    x = Conv1D(3, 3, activation='relu')(input_ecg)
    for i in range(8):
        x = MaxPool1D(2)(x)
        x = Conv1D(32, 5, activation='relu')(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="sigmoid")(x)
    ret_model = Model(input_ecg, output)

    return ret_model


def fit_save(model, x, y, batch_size, validation_data, epochs, repeat=15, name="model"):
    counter = 0
    max_se = 0
    max_sp = 0
    class_weight = {0: y[:,0].sum(),
                    1: y[:,1].sum()
                    }
    for epoch in np.arange(0, epochs):
        counter += 1
        model.fit(x, y, batch_size=batch_size, epochs=1,
                  validation_data=validation_data, verbose=0, class_weight=class_weight)

        y_prediction = np.argmax(model.predict(validation_data[0]), axis=1)
        y_labels = np.argmax(validation_data[1], axis=1)

        tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)
        if se + sp > max_se + max_sp:
            max_se = se
            max_sp = sp
            model.save('./models\\' + name + ".h5")
            counter = 0

        print('epoch: ' + str(epoch) + ', se: ' + str(se.round(4)) + ', sp: ', str(sp.round(4))
              + ", max se: " + str(max_se.round(4)) + ", max sp: " + str(max_sp.round(4)))

        if counter > repeat:
            break


diags_norm_rythm = [0]
diags_fibrilation = [15]
diags_flutter = [16, 17, 18]
diags_hypertrophy = [119, 120]
diags_extrasystole = [69, 70, 71, 72, 73, 74, 75, 86]

diags_fibrilation_old = [140]


def evaluate_model(model, diagnoses, epochs, batch_size, rnd=42, name='model'):
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_split(diagnoses, rnd_state=rnd, is_new=True)

    fit_save(model, X_train, Y_train, batch_size, (X_val, Y_val), epochs, name=name)

    model = load_model('./models\\' + name + ".h5")
    y_prediction = np.argmax(model.predict(X_test), axis=1)
    y_labels = np.argmax(Y_test, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_labels, y_prediction).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    print('Test se: ', str(se) + ', sp: ', str(sp))


if __name__ == "__main__":
    model = build_model1(DATA_SIZE)
    sgd_opt = SGD(0.001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])
    model.summary()

    evaluate_model(model, diags_fibrilation, 1000, 64)
