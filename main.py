from os import listdir, path
from os.path import isfile, join
from re import compile
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import nanmean, nan
from genetic import Population
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt


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
    X_dense[X_dense == 0] = nan

    # Use nanmean of numpy which will ignore nan while calculating the mean
    means = nanmean(X_dense, axis=0)

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
        if sum(agent.value) >= 1000:
            agent.fitness = sum(chromo_tfidf_words)/sum(agent.value)*1000
        elif sum(agent.value) >= 2000 or sum(agent.value) < 1000:
            agent.fitness = sum(chromo_tfidf_words)/(sum(agent.value)+200)*1000
    return agents


def ga(POP_SIZE, BITS, PC, PM, generations):
    
    # get values to use
    TF_IDF_means = calc_tdif_means()
    # variable to check the number of iters that algorithm is not improving
    cnt = 0

    fit = []
    performance = []
    for _ in range(3):
        # create population
        population = Population(POP_SIZE, BITS, PC, PM)

        for i in range(generations):   

            # apply fitness function to them
            population.population = fitness(population.population, TF_IDF_means)[:]
            # get fittest agent 
            best_agent = population.get_fittest()
            fit.append(best_agent.fitness)
            print(f'Generation: {i} [Local Best: {sum(best_agent.value)} --> {best_agent.fitness}]')

            # check termination conditions
            # or 
            if best_agent.fitness <= fit[i-1] or fit[i-1]*1.01 > best_agent.fitness:
                cnt += 1
            else:
                cnt = 0
            if cnt >= 80:
                break

            # apply genetic operators
            population.selection(select='rank_roullete')
            population.crossover(select='multi', N=50)
            population.mutation()

        performance.append(fit)
        fit = []
        cnt = 0

    else:
        return performance


if __name__ == '__main__':

    POP_SIZE, BITS, PC, PM, GENS = 200, 8520, 0.6, 0.01, 500
    per = ga(POP_SIZE, BITS, PC, PM, GENS)

    # average ages
    avg_ages = sum([len(subarr) for subarr in per])/3
    print(avg_ages)
    # find longest 
    max_l = len(max(per, key=len))
    # apply padding just in case
    arr = numpy.array(([subarr + [subarr[-1]] * (max_l - len(subarr)) for subarr in per]))
    # average them
    avg_per_age = arr.mean(axis=0)
    plt.plot(avg_per_age)
    plt.show()
    
    # to set varianve to initial population creation
    # αναπροσαρμογη πιθανοτητας ισως
    # mutation with elitism
