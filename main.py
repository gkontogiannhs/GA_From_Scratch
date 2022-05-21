from os import listdir
from os import path
from os.path import isfile, join
from re import findall, compile, sub
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from genetic import ga


# main path
home = path.expanduser('~')
mypath = f'{home}/Downloads/DeliciousMIL/Data/'
# list files
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def get_documents(filename):
    with open(filename, encoding='utf-8') as f:
        corpus = []
        regex = compile(r'<(.*?)>')
        
        documents = f.readlines()
        for doc in documents:
            list_doc = doc.strip('\n').split(' ')
            corpus.append(' '.join(word for word in list_doc if not regex.match(word)))
            
        return corpus

corpus = get_documents(mypath+files[3])

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)

X_dense = X.todense()
# TFIDF of words not in the doc will be 0, so replace them with nan
X_dense[X_dense == 0] = np.nan
# Use nanmean of numpy which will ignore nan while calculating the mean
means = np.nanmean(X_dense, axis=0)
# convert it into a dictionary for later lookup
TF_IDF_means = dict(zip(tfidf_vectorizer.get_feature_names(), means.tolist()[0]))

# not all words are present in the dataset
# so we append missing with tfidf value of 0
for key in range(8520):
    if str(key) not in TF_IDF_means:
        TF_IDF_means[str(key)] = 0


sum(TF_IDF_means.values())/8510*1000

for _ in range(100):
    chromosome = np.random.choice([0, 1], size=(X.shape[1],), p=[0.82, 0.18])
    chromo_tfidf_words = [TF_IDF_means[str(i)] for i in range(chromosome.shape[0]) if chromosome[i] == 1]
    tfidf_mean = np.sum(chromo_tfidf_words)/np.sum(chromosome)*1000
    print(tfidf_mean)



if __name__ == '__main__':
    POP_SIZE, BITS, PC, PM, GENS = 5, 5, 0.25, 0.01, 2000
    ga(POP_SIZE, BITS, PC, PM, GENS)