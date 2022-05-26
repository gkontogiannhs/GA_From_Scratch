from numpy import cumsum
from numpy.random import choice, rand
from random import randint, random, uniform, choices

class Agent():
    def __init__(self, length):
        self.length = length
        self.value = choice([0, 1], size=(self.length,), p=[7/8, 1/8]).tolist()
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

         
    def __str__(self):
        string = ''
        for agent in self.population:
            string += 'Value: ' + str(agent.value) + ' Fitness: ' + str(agent.fitness) + '\n'
        return string
    
    def cost_roullete(self):
        # total fitness
        total_fitness = sum(a.fitness for a in self.population)
        # probs of each agent
        probs = [a.fitness/total_fitness for a in self.population]
        # accumulative probs
        q_probs = cumsum(probs).tolist()
    
        new_gen = []
        # spin roullete pop_size times
        for _ in range(self.pop_size):
            # generate prob
            p = uniform(0, 1)
            # find correct slot
            for j in range(self.pop_size):
                if p <= q_probs[j]:
                    new_gen.append(self.population[j])
                    break
        
        self.population = []
        self.population = new_gen[:]
        return self

    def rank_roullete(self, q_probs=[]):
        # sort by best
        agents = sorted(self.population, key=lambda a: a.fitness, reverse=True)

        denom = sum(range(1, self.pop_size))
        q_probs = cumsum([(self.pop_size-i+1)/denom for i in range(1, self.pop_size+1)]).tolist()
        
        new_gen = []
        # spin roullete pop_size times
        for _ in range(self.pop_size):
            # generate prob
            p = uniform(0, 1)
            # find correct slot
            for j in range(self.pop_size):
                if p <= q_probs[j]:
                    new_gen.append(agents[j])
                    break
        self.population = []
        self.population = new_gen[:]
        return self

    
    def tournament(self):
        new_gen = []
        for _ in range(self.pop_size):
            fight = choices(self.population, k=randint(2, self.pop_size//2))
            best_agent = sorted(fight, key=lambda a: a.fitness, reverse=True)[0]
            new_gen.append(best_agent)

        self.population = new_gen[:]

    def selection(self, select='cost_roullete'):
        if select == 'cost_roullete':
            self.cost_roullete()
        elif select == 'rank_roullete':
            self.rank_roullete()
        elif select == 'tournament':
            self.tournament()
        else:
            raise('Not valid option!')
    
    def single_point_cross(self):
        offspring = []

        for i in range(0, self.pop_size, 2):

            if  random() < self.pc:

                # pick pairs
                bs1, bs2 = self.population[i], self.population[i+1]
                
                # random index to slice
                pt = randint(1, self.bits-2)

                # create ancestors
                child1 = Agent(self.bits)
                child2 = Agent(self.bits)
                
                # give them the values
                child1.value = bs1.value[:pt] + bs2.value[pt:]
                child2.value = bs2.value[:pt] + bs1.value[pt:]
                
                # append offsprings
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(self.population[i])
                offspring.append(self.population[i+1])

        self.population = offspring[:]

    def multi_point_cross(self, N):
        for _ in range(N):
            self.single_point_cross()

    def uniform_cross(self):

        offspring = []
        for i in range(0, self.pop_size, 2):

            # pick pairs
            bs1, bs2 = self.population[i], self.population[i+1]
            if  random() < self.pc:
                # create ancestors
                child1 = Agent(self.bits)
                child2 = Agent(self.bits)
                
                probs = rand(self.pop_size)
                for i in range(len(probs)):
                    if probs[i] < 0.5:
                        temp = bs1.value[i]
                        bs1.value[i] = bs2.value[i]
                        bs2.value[i] = temp

                # give them the values
                child1.value = bs1.value
                child2.value = bs2.value
                
                # append offsprings
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(self.population[i])
                offspring.append(self.population[i+1])

        self.population = offspring[:]

    def crossover(self, select='single', N=50):    
        if select == 'single':
            self.single_point_cross()
        elif select == 'multi':
            for _ in range(N):
                self.single_point_cross()
        elif select == 'uniform':
            self.uniform_cross()
        else:
            raise('Not valid selection option!')
        return self
    
    def mutation(self):
        best = self.get_fittest()
        for agent in self.population:
            if agent != best:
                for i in range(self.bits):
                    if random() <= self.pm:
                        agent.value[i] = 1 - agent.value[i]
        return self
            
    def get_fittest(self):
        index = 0
        for i in range(1, len(self.population)):
            if self.population[i].fitness > self.population[index].fitness:
                index = i
        return self.population[index]