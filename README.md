# This repo hosts an object oriented implementation of a Genetic Algorithm trying to find the most important words in corpus in order to be fed in a neural network
## genetic.py contains the code for it accompanied by the basic mutation method and some crossover, selection methods based on different teqnuiches  
### For the procedure of selection, suit your needs by picking between:
- cost roullete selection
- rank roullete selection
- tournament
### For the procedure of crossover:
- single-point crossover
- multi-point crossover
- uniform crossover

#### The main program (main.py) calls the genetic algorithm constructor and applies it in order to to find the 1000 most important words in a corpus, based on multiple documents. This is done by creating a fitness function based on the TF-IDF metric. When the algorithm converges, it stores it's solution in order to remove the words from the documents that are not present in that (corpus) solution. Neural network model is now fed again with the reduced document sizes aiming for improved accuracy. Neural Network model built and training can be found in NN.py
