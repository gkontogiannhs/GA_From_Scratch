from main import get_documents
from os import path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def get_y(filename):
    
    try:
        with open(filename) as y_file:
            y = []
            while True:
                # get each line from the files
                label_line = y_file.readline().strip('\n')
                # check if EOF
                if len(label_line) == 0:
                    return np.asarray(y, dtype=int)
                # labels
                y.append(label_line.split(' '))

    except FileNotFoundError:
        raise "Count not read file"


def get_model(n_inputs, n_outputs, n_hidden1, lr, m):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    model.add(keras.layers.Dense(n_hidden1, activation='relu'))
    model.add(keras.layers.Dense(n_outputs, activation='sigmoid'))
    opt = keras.optimizers.SGD(learning_rate=lr, momentum=m)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    return model

def evaluate_model(X, y):
    train_acc, test_acc, train_loss, test_loss = [], [], [], []
    # train - test data split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the data to training and testing data 5-Fold
    kfold = KFold(n_splits=2, shuffle=True)
    for i, (train, eval) in enumerate(kfold.split(X_train)):

        # create model
        # input-demensions, output-dims, loss-function, hidden1, learning-rate
        model = get_model(X.shape[1], y.shape[1], 20, 0.01, 0.6)

        # Fit model
        h = model.fit(X_train[train], y_train[train], validation_data=(X_train[eval], y_train[eval]), epochs=150, verbose=0)

        # store for each fold the history
        train_acc.append(h.history['acc'])
        test_acc.append(h.history['val_acc'])
        train_loss.append(h.history['loss'])
        test_loss.append(h.history['val_loss'])

        # make predict to unseen data
        yhat = model.predict(X_test)    
        yhat = yhat.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)

    # average folds
    train_acc = np.average(train_acc, axis=0)
    test_acc = np.average(test_acc, axis=0)
    train_loss = np.average(train_loss, axis=0)
    test_loss = np.average(test_loss, axis=0)
    
    # plot averaged folds
    plot(train_acc, test_acc, train_loss, test_loss, 'CE')

    
def plot(train_acc, test_acc, train_loss, test_loss, label):
    # summarize history for loss
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title(label + ' - Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title(label + ' - Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
if __name__ == '__main__':

    # get corpus
    home = path.expanduser('~')
    mypath = f'{home}/Downloads/DeliciousMIL/Data/'
    corpus = get_documents(mypath+'train-data.dat')

    # load the best solution that genetic algorith found
    try:
        with open('best_solution.txt', encoding='utf-8') as file:
            string_vocab = file.readline().split(' ')
            # convert indexes to integer words
            new_vocab = [str(i) for i in range(len(string_vocab)) if string_vocab[i] == '1']

    except FileNotFoundError:
        raise('File is not found. Make sure that the file is saved locally.')

    # now what we need is to parse documents from corpus and remove words that are 
    # not predent in new_vocab
    new_corpus = []
    for document in corpus:
        doc = document.split(' ')
        new_corpus.append(' '.join(word for word in doc if word in new_vocab))

    # get y labels
    y = get_y(mypath+'train-label.dat')

    # get BoW
    X_bow = CountVectorizer().fit_transform(new_corpus).toarray()

    ################ NORMALIZATION ########################################
    X = MinMaxScaler().fit_transform(X_bow)

    evaluate_model(X, y)