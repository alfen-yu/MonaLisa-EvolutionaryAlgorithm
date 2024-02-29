from evolutionary_algorithm import *
import random

class SelectionSchemes:
    """
    This class contains methods for various selection schemes to be implemented for our problems. The selection schemes will be used in the Evolutionary Algorithms as well for the parent selection methods and evaluating the fitness of our population. 
    The selection schemes include:
        - Fitness Proportional Selection
        - Rank Based Selection
        - Binary Tournament
        - Truncation
        - Random
    
    * Note: The selection schemes will be implemented as static methods. 
    """

    @staticmethod
    def fitnessProportional(population: list, fitnessVals: list, selectCount: int) -> list:
        """
        This method implements the fitness proportional selection scheme.

        Args:
            - population (list): The population of individuals to be selected from.
            - fitnessVals (list): The fitness values of the individuals in the population.
            - selectCount (int): The number of individuals to be selected from the population.
        
        Returns:
            - selected (list): The list of selected individuals.
        """
        # print("Fitness Proportional Selection")
        if(len(population) != len(fitnessVals)):
            raise Exception("Population and fitness values must be of the same length.")
        if(selectCount > len(population)):
            raise Exception("The number of individuals to be selected cannot be greater than the population size.")
        total = sum(fitnessVals)
        probabilities = [f/total for f in fitnessVals]
        return random.choices(population, weights=probabilities, k=selectCount)

    @staticmethod
    def rankBased(population: list, fitnessVals: list, selectCount: int) -> list:
        """
        This method implements the rank based selection scheme. The fitness values are normalized to create ranges which act as ranks for the individuals in the population. The individuals are then selected based on their rank.

        Args:
            - population (list): The population of individuals to be selected from.
            - fitnessVals (list): The fitness values of the individuals in the population.
            - selectCount (int): The number of individuals to be selected from the population.
        
        Returns:
            - selected (list): The list of selected individuals.
        """
        # print("Rank Based Selection")
        if(len(population) != len(fitnessVals)):
            raise Exception("Population and fitness values must be of the same length.")
        if(selectCount > len(population)):
            raise Exception("The number of individuals to be selected cannot be greater than the population size.")
        
        sorted_indices = sorted(range(len(fitnessVals)), key=lambda k: fitnessVals[k])
        ranks = [sorted_indices.index(i) + 2 for i in range(len(fitnessVals))]
        normalized_ranks = [r / len(ranks) for r in ranks]
        return random.choices(population, weights=normalized_ranks, k=selectCount)

    @staticmethod
    def binaryTournament(population: list, fitnessVals: list, selectCount: int) -> list:
        """
        This method implements the binary tournament selection scheme. The individuals are selected in pairs (or a set value) and the individual with the highest fitness is selected. In this implementation, an elitism approach is used to ensure that the best individual is always sent forward as well.

        Args:
            - population (list): The population of individuals to be selected from.
            - fitnessVals (list): The fitness values of the individuals in the population.
            - selectCount (int): The number of individuals to be selected from the population.

        Returns:
            - selected (list): The list of selected individuals.
        """
        # print("Binary-Tournament Selection")
        if(len(population) != len(fitnessVals)):
            raise Exception("Population and fitness values must be of the same length.")
        if(selectCount > len(population)):
            raise Exception("The number of individuals to be selected cannot be greater than the population size.")
        #Elitism
        selected = []
        bestIndiv = max(zip(population, fitnessVals), key=lambda x: x[1])[0]
        for _ in range(selectCount - 1):
            candidates = random.sample(range(len(population)), 2)
            selected.append(population[max(candidates, key=lambda x: fitnessVals[x])])
        
        selected.append(bestIndiv)
        if len(selected) != selectCount:
            raise Exception("The number of selected individuals does not match the selectCount.")
        return selected
    
    @staticmethod
    def truncation(population: list, fitnessVals: list, selectCount: int) -> list:
        """
        This method selects chromosomes with the highest fitness values. The number of chromosomes selected is based on the selectCount parameter.

        Args:
            - population (list): The population of individuals to be selected from.
            - fitnessVals (list): The fitness values of the individuals in the population.
            - selectCount (int): The number of individuals to be selected from the population.
        
        Returns:
            - selected (list): The list of selected individuals.
        """
        # print("Truncation Selection")
        if(len(population) != len(fitnessVals)):
            raise Exception("Population and fitness values must be of the same length.")
        if(selectCount > len(population)):
            raise Exception("The number of individuals to be selected cannot be greater than the population size.")
        
        indices = list(range(len(population)))
        indices.sort(key=lambda x: fitnessVals[x], reverse=True)
        return [population[i] for i in indices[:selectCount]]


    @staticmethod
    def random(population: list, fitnessVals: list, selectCount: int) -> list:
        """
        This method randomly selects individuals from the population.

        Args:
            - population (list): The population of individuals to be selected from.
            - fitnessVals (list): The fitness values of the individuals in the population.
            - selectCount (int): The number of individuals to be selected from the population.
        
        Returns:
            - selected (list): The list of selected individuals.
        """
        # print("Random Selection")
        if(len(population) != len(fitnessVals)):
            raise Exception("Population and fitness values must be of the same length.")
        if(selectCount > len(population)):
            raise Exception("The number of individuals to be selected cannot be greater than the population size.")
        return random.sample(population, selectCount)