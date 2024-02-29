from problem import Problem
from selection_schemes import SelectionSchemes

# Global Params - Initial
# Population_Size = 30
# Num_Offsprings = 10
# Num_Generations = 50
# Mutation_Rate = 0.5
# Iterations = 10

# Global Params - Optimalish
Population_Size = 100
Num_Offsprings = 100
Num_Generations = 100
Mutation_Rate = 0.7
Iterations = 5

class EvolutionaryAlgorithm:
    """
    This class will be responsible for the evolutionary cycle, including the parent selection, and survival selection implementations.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.population = self.problem.population
        self.fitnessVals = self.problem.fitnessVals
        self.generation = 1
    
    def getBestFitnessScore(self):
        """ Returns the best Fitness Score """
        return max(self.fitnessVals)

    def getBestIndiv(self):
        """ Returns the best individual """
        return self.population[self.fitnessVals.index(self.getBestFitnessScore())]

    def getWorstFitnessScore(self):
        """ Returns the worst Fitness Score """
        return min(self.fitnessVals)        

    def getWorstIndiv(self):
        """ Returns the worst individual """
        return self.population[self.fitnessVals.index(self.getWorstFitnessScore())]

    def getAvgFitnessScore(self):
        """ Returns the average Fitness Score """
        return sum(self.fitnessVals)/len(self.fitnessVals)

    def ParentSelection(self, selection: str) -> None:
        """
        Generates an offspring from the population, based on the selection method used. 
        Args:
            - selection: The selection method to be used as a string

        Returns:
            - None 
        """
        selection_method = getattr(SelectionSchemes, selection)
        parents = selection_method(self.population, self.fitnessVals, Num_Offsprings)
        for i in range(0, Num_Offsprings, 2):
            bacha1, bacha2 = self.problem.crossover(parents[i], parents[i+1]), self.problem.crossover(parents[i], parents[i+1])
            self.population.append(self.problem.mutate(Mutation_Rate, bacha1))
            self.population.append(self.problem.mutate(Mutation_Rate, bacha2))


    def SurvivalSelection(self, selection: str, selectCount: int) -> None:
        """
        This method will evaluate the chromosomes in our population based on the selection method used. An elitism approach is used inherently for survival selection. This elitism approach was used after a trial and error, and it was found that the elitism approach was effective overall inherently. 
        Args:
            - selection: The selection method to be used as a string
            - selectCount: The number of individuals to be selected from the population.

        Returns:
            - None 
        """
        self.problem.population = self.population
        self.fitnessVals = self.problem.fitness()

        selection_method = getattr(SelectionSchemes, selection)
        newgen = selection_method(self.population, self.fitnessVals, selectCount)

        bestIndiv = self.getBestIndiv()
        newgen.append(bestIndiv)

        self.population = newgen
        self.problem.population = self.population
        self.fitnessVals = self.problem.fitness()
        self.generation += 1
