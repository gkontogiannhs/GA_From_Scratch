from main import get_documents
from os import path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


"""
before_after = []
    from pandas import DataFrame
    from matplotlib import pyplot as plt
    for i, doc in enumerate(corpus):
        # print(f'Document {i}|  Before: {len(doc.split(" "))} |  After: {len(new_corpus[i].split(" "))}')
        before_after += [(len(doc.split(" ")), len(new_corpus[i].split(" ")))]

    df = DataFrame(before_after, columns=['Before', 'After'])
    df.plot()
    plt.show()
"""

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

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model(n_inputs, n_outputs, loss_f, n_hidden1, n_hidden2, lr, m, wd):
    model = keras.models.Sequential()
    model.add(keras.Input(shape=(n_inputs,)))
    model.add(keras.layers.Dense(n_hidden1, activation='relu'))
    # model.add(keras.layers.Dense(n_hidden2, activation='relu'))
    model.add(keras.layers.Dense(n_outputs, activation='sigmoid'))

    opt = keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss_f)

    return model

def evaluate_model(X, y):
    train_acc, test_acc, train_loss, test_loss = [], [], [], []

    # train - test data split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create model
    # input-demensions, output-dims, loss-function, hidden1, hidden2, learning-rate, momentum, weight-decay
    model = get_model(X.shape[1], y.shape[1], 'binary_crossentropy', 5, 20, 0.01, 0.6, 0.1)

    # Fit model
    h = model.fit(X_train, y_train, validation_split=0.2, epochs=300, verbose=0)

    # store for each fold the history
    #train_acc.append(h.history['binary_accuracy'])
    #test_acc.append(h.history['val_binary_accuracy'])
    #train_loss.append(h.history['loss'])
    #test_loss.append(h.history['val_loss'])

    # plot averaged folds
    # plot(train_acc[0], test_acc[0], train_loss[0], test_loss[0], 'RMSE')
        
    # make predict to unseen data
    yhat = model.predict(X_test)    
    yhat = yhat.round()
    acc = accuracy_score(y_test, yhat)
    # store result
    print('>%.3f' % acc)
    print(classification_report(y_test, yhat))

    
def plot(train_acc, test_acc, train_loss, test_loss, label):
    plt.plot(train_acc)
    plt.plot(test_acc)
    plt.title(label + ' - Accuracy')
    plt.ylabel('accuracy')
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

    # tranform X, y to numpy arrays
    X_bow = TfidfVectorizer().fit_transform(new_corpus).toarray()

    ################ NORMALIZATION ########################################
    X = MinMaxScaler().fit_transform(X_bow)

    evaluate_model(X, y)