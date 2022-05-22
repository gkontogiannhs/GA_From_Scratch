from numpy import cumsum
from numpy.random import choice
from random import randint, random, uniform

class Agent():
    def __init__(self, length):
        self.length = length
        self.value = choice([0, 1], size=(self.length,)).tolist()
        self.fitness = -1
        
    def __str__(self):
        return 'Value: ' + str(self.value) + ' Fitness: ' + str(self.fitness) + '\n'

class Population():
    def __init__(self, pop_size, bits, pc, pm):
        self.pop_size = pop_size
        self.bits = bits
        self.pc = pc
        self.pm = pm
        self.population = [Agent(self.bits) for _ in range(self.pop_size)]
        self.newGeneration = []

         
    def __str__(self):
        string = ''
        for agent in self.population:
            string += 'Value: ' + str(agent.value) + ' Fitness: ' + str(agent.fitness) + '\n'
        return string
    
    def selection(self):
        # sort by best
        # agents = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        agents = self.population[:]
        # total fitness
        total_fitness = sum(a.fitness for a in agents)
        # probs of each agent
        probs = [a.fitness/total_fitness for a in agents]
        # accumulative probs
        q_probs = cumsum(probs).tolist()
        
        # spin roullete pop_size times
        for _ in range(self.pop_size):
            # generate prob
            p = uniform()
            # find correct slot
            for j in range(self.pop_size):
                if p <= q_probs[j]:
                    self.newGeneration.append(agents[j])
                    break
        self.population = self.newGeneration[:]
        return self
    
    def crossover(self):    
        crossovers = []
        offspring = []
        
        for agent in self.population:
            # probability of cross over
            if random() <= self.pc:
                crossovers.append(agent)
        
        cross_len = len(crossovers)
        if cross_len > 1:
            for i in range(cross_len // 2):
                
                # random index to slice
                pt = randint(0, self.bits-1)
                
                # pick pairs
                bs1 = crossovers[i]
                bs2 = crossovers[i+1]
                
                if bs1 != bs2:
                    # create ancestors
                    child1 = Agent(self.bits)
                    child2 = Agent(self.bits)
                    
                    # give them the values
                    child1.value = bs1.value[:pt] + bs2.value[pt:]
                    child2.value = bs2.value[:pt] + bs1.value[pt:]
                    
                    # append offsprings
                    offspring.append(child1)
                    offspring.append(child2)
                    
            # remove parents
            for cro in crossovers:
                self.population.remove(cro)
                
            self.population.extend(offspring)
        
        return self
    
    def mutation(self):
        for agent in self.population:
            for i in range(self.bits):
                if random() <= self.pm:
                    agent.value[i] = 1 - agent.value[i]
        return self
    
    def fitness(self):
        for a in self.population:
            a.fitness = sum(a.value)
        return self

    def validate_agent(self, agent):
        if agent.fitness >= 9:
            return True
        return False
            
    def find_best(self):
        index = 0
        for i in range(1, len(self.population)):
            if self.population[i].fitness > self.population[index].fitness:
                index = i
        return self.population[index]

def ga(POP_SIZE, BITS, PC, PM, generations):
    
    # create population
    population = Population(POP_SIZE, BITS, PC, PM)

    for i in range(generations):   
        
        # apply fitness function to them
        population.fitness()
        
        best_agent = population.find_best()
        print('Generation: '+ str(i)+ '   {Local Best: '+ str(sum(best_agent.value)) + ' --> '+ str(best_agent.fitness) + '}')

        if population.validate_agent(best_agent):
            print('\nWe got winner:')
            print(best_agent.fitness)
            return best_agent

        # apply genetic operators
        population.selection()
        population.crossover()
        population.mutation()


if __name__ == '__main__':

    POP_SIZE, BITS, PC, PM, GENS = 5, 10, 0.25, 0.01, 1000
    best_agent = ga(POP_SIZE, BITS, PC, PM, GENS)
    print(best_agent)