from os import listdir
from os import path
from os.path import isfile, join
from re import compile
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from genetic import Population


def get_documents(filename):
    with open(filename, encoding='utf-8') as f:
        corpus = []
        regex = compile(r'<(.*?)>')
            
        documents = f.readlines()
        for doc in documents:
            list_doc = doc.strip('\n').split(' ')
            corpus.append(' '.join(word for word in list_doc if not regex.match(word)))
                
        return corpus

def calc_tdif_means():

    # main path
    home = path.expanduser('~')
    mypath = f'{home}/Downloads/DeliciousMIL/Data/'
    # list files
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    corpus = get_documents(mypath+files[3])

    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(corpus)

    X_dense = X.todense()

    # TFIDF of words not in the doc will be 0, so replace them with nan
    X_dense[X_dense == 0] = np.nan

    # Use nanmean of numpy which will ignore nan while calculating the mean
    means = np.nanmean(X_dense, axis=0)

    # convert it into a dictionary for later lookup
    TF_IDF_means = dict(zip(tfidf_vectorizer.get_feature_names_out(), means.tolist()[0]))

    # not all words are present in the dataset
    # so we append missing with tfidf value of 0
    for key in range(8520):
        if str(key) not in TF_IDF_means:
            TF_IDF_means[str(key)] = 0

    return TF_IDF_means


def fitness(agents, TF_IDF_means):
    for agent in agents:
        chromo_tfidf_words = [TF_IDF_means[str(i)] for i in range(BITS) if agent.value[i] == 1]
        if np.sum(agent.value) >= 1000:
            agent.fitness = np.sum(chromo_tfidf_words)/np.sum(agent.value)*1000
        else:
            print('Invalid solution!')
            agent.fitness = np.sum(chromo_tfidf_words)/(agent.length-np.sum(agent.value))*1000
    return agents


def ga(POP_SIZE, BITS, PC, PM, generations):
    
    # create population
    population = Population(POP_SIZE, BITS, PC, PM)

    # get values to use
    TF_IDF_means = calc_tdif_means()

    for i in range(generations):   
        
        # apply fitness function to them
        population.population = fitness(population.population, TF_IDF_means)
        
        best_agent = population.find_best()
        print('Generation: '+ str(i)+ '   {Local Best: '+ str(np.sum(best_agent.value)) + ' --> '+ str(best_agent.fitness) + '}')

        if population.validate_agent(best_agent):
            print('\nWe got winner:')
            print(best_agent.fitness)
            break

        # apply genetic operators
        population.selection()
        population.crossover()
        population.mutation()


if __name__ == '__main__':

    POP_SIZE, BITS, PC, PM, GENS = 10, 8520, 0.25, 0.01, 100
    ga(POP_SIZE, BITS, PC, PM, GENS)
