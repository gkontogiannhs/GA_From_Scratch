#!/usr/bin/env python
# coding: utf-8
# Κωδικοποίηση (Coding)
# Αρχικοποίηση (Initialization)
# Αποκωδικοποίηση (Decoding)
# Υπολογισμός ικανότητας ή αξιολόγηση (Fitness calculation ή evaluation)
# Επιλογή (Selection)
# Αναπαραγωγή (Reproduction)
# Διασταύρωση (Crossover ή mating)
# Μετάλλαξη (Mutation)
# Επανάληψη από το βήμα (2) μέχρι να ικανοποιηθεί το κριτήριο τερματισμού του Γ.Α
# In[224]:


import numpy as np
import random

class Agent():
    def __init__(self, length):
        self.length = length
        self.value = np.random.choice([0, 1], size=(self.length,), p=[1/2, 1/2])
        self.fitness = -1
        
    def __str__(self):
        return 'Value: ' + str(self.value) + ' Fitness: ' + str(self.fitness)
    
    # convert to decimal
    def decode(self):
        value = 0
        pow = self.length-1
        for bit in self.value:
            value += (bit * (2**pow))
            pow -= 1
            
        # self.value = value
        return value
        
class GA():
    def __init__(self, pop_size, bits, pc, pm):
        self.population = []
        self.pop_size = pop_size
        self.bits = bits
        self.pc = pc
        self.pm = pm
        # self.newGeneration = []
    
    # call create chromosome pop_size times
    def create_population(self):
        # population will be a list of Agent Objects
        self.population = [Agent(self.bits) for _ in range(self.pop_size)]
        return self.population
    
    def fitness(self, agents):
        for agent in agents:
            agent.fitness = agent.decode() ** 2
        return agents
    
        
    def selection(self, agents):
        # sort by best
        agents = sorted(agents, key=lambda a: a.fitness, reverse=True)
        # total fitness
        total_fitness = sum(a.fitness for a in agents)
        # probs of each agent
        probs = [a.fitness/total_fitness for a in agents]
        # accumulative probs
        q_probs = np.cumsum(probs)
    
        selected_agents = []
        
        # spin roullete pop_size times
        for _ in range(len(agents)):
            p = random.random()
            # find correct slot
            for j in range(len(agents)):
                if p <= q_probs[j]:
                    # append 
                    selected_agents.append(agents[j])
                    break
        
        return selected_agents
    
    def crossover(self, agents):    
        crossovers = []
        offspring = []
        for agent in agents:
            # probability of cross over
            if random.random() <= self.pc:
                crossovers.append(agent)
    
        if len(crossovers) > 1:
            for i in range(int(len(crossovers)/2)):
                
                # random index to slice
                pt = random.randint(0, self.bits-1)

                bs1 = crossovers[i]
                bs2 = crossovers[i+1]

                # create ancestors
                child1 = Agent(self.bits)
                child2 = Agent(self.bits)

                # give them the values
                child1.value = np.concatenate((bs1.value[:pt], bs2.value[pt:]))
                child2.value = np.concatenate((bs2.value[:pt], bs1.value[pt:]))

                # append offsprings
                offspring.append(child1)
                offspring.append(child2)

                # remove parents
                agents.remove(bs1)
                agents.remove(bs2)
                
            agents.extend(offspring)
        
        return agents
    
    def mutation(self, agents):
        for agent in agents:
            for i, bit in enumerate(agent.value):
                if random.random() <= self.pm:
                    if bit == 0:
                        agent.value[i] = 1
                    else:
                        agent.value[i] = 0
                        
        return agents
    
    def validate_agent(self, agent):
        if agent.fitness >= 841:
            return True
        return False
            
    def find_best(self, agents):
        index = 0
        for i in range(1, len(agents)):
            if agents[i].fitness > agents[index].fitness:
                index = i
        return agents[index]
            
    
    def fit(self, generations):
        # create population
        agents = self.create_population().copy()
        # apply fitness function to them
        agents = self.fitness(agents).copy()
        
        for i in range(generations):   
            
            best_agent = self.find_best(agents)
            print('Generation: '+ str(i)+ '   {Local Best: '+ str(best_agent.value) + ' --> '+ str(best_agent.decode()) + '}')
            
            if self.validate_agent(best_agent):
                print('We got winner:')
                print(best_agent.fitness)
                break
                
            # apply genetic operators
            agents = self.selection(agents).copy()
            agents = self.crossover(agents).copy()
            agents = self.mutation(agents).copy()
            
            # re calculate fitnesses
            agents = self.fitness(agents).copy()


# In[225]:


if __name__ == '__main__':
    POP_SIZE, BITS, PC, PM = 5, 5, 0.25, 0.01
    ga = GA(POP_SIZE, BITS, PC, PM)
    ga.fit(100)


# In[ ]:




