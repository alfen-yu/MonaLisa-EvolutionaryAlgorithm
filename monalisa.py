from abc import ABC, abstractmethod
from typing import List, Tuple
import random
from PIL import Image, ImageDraw
import numpy as np

class SelectionSchemes:
    """
    This class contains methods for various selection schemes to be implemented for our problems. 
    The selection schemes will be used in the Evolutionary Algorithms as well for the parent 
    selection methods and evaluating the fitness of our population. 
    The selection schemes include:
        - Fitness Proportional Selection
        - Rank Based Selection
        - Binary Tournament
        - Truncation
        - Random
    
    * Note: The selection schemes will be implemented as static methods. 
    """

    @staticmethod
    def fitnessProportional(population: List, fitnessVals: List, selectCount: int) -> List:
        """
        This method implements the fitness proportional selection scheme.
        """
        if len(population) != len(fitnessVals):
            raise ValueError("Population and fitness values must be of the same length.")
        if selectCount > len(population):
            raise ValueError("The number of individuals to be selected cannot be greater than the population size.")
        total = sum(fitnessVals)
        probabilities = [f / total for f in fitnessVals]
        return random.choices(population, weights=probabilities, k=selectCount)

    @staticmethod
    def rankBased(population: List, fitnessVals: List, selectCount: int) -> List:
        """
        This method implements the rank based selection scheme. The fitness values are normalized 
        to create ranges which act as ranks for the individuals in the population. 
        The individuals are then selected based on their rank.
        """
        if len(population) != len(fitnessVals):
            raise ValueError("Population and fitness values must be of the same length.")
        if selectCount > len(population):
            raise ValueError("The number of individuals to be selected cannot be greater than the population size.")
        
        sorted_indices = sorted(range(len(fitnessVals)), key=lambda k: fitnessVals[k])
        ranks = [sorted_indices.index(i) + 2 for i in range(len(fitnessVals))]
        normalized_ranks = [r / len(ranks) for r in ranks]
        return random.choices(population, weights=normalized_ranks, k=selectCount)

    @staticmethod
    def binaryTournament(population: List, fitnessVals: List, selectCount: int) -> List:
        """
        This method implements the binary tournament selection scheme. The individuals are selected 
        in pairs (or a set value) and the individual with the highest fitness is selected. 
        In this implementation, an elitism approach is used to ensure that the best individual is 
        always sent forward as well.
        """
        if len(population) != len(fitnessVals):
            raise ValueError("Population and fitness values must be of the same length.")
        if selectCount > len(population):
            raise ValueError("The number of individuals to be selected cannot be greater than the population size.")
        
        selected = []
        bestIndiv = max(zip(population, fitnessVals), key=lambda x: x[1])[0]
        for _ in range(selectCount - 1):
            candidates = random.sample(range(len(population)), 2)
            selected.append(population[max(candidates, key=lambda x: fitnessVals[x])])
        
        selected.append(bestIndiv)
        if len(selected) != selectCount:
            raise ValueError("The number of selected individuals does not match the selectCount.")
        return selected
    
    @staticmethod
    def truncation(population: List, fitnessVals: List, selectCount: int) -> List:
        """
        This method selects chromosomes with the highest fitness values. 
        The number of chromosomes selected is based on the selectCount parameter.
        """
        if len(population) != len(fitnessVals):
            raise ValueError("Population and fitness values must be of the same length.")
        if selectCount > len(population):
            raise ValueError("The number of individuals to be selected cannot be greater than the population size.")
        
        indices = sorted(range(len(fitnessVals)), key=lambda x: fitnessVals[x], reverse=True)
        return [population[i] for i in indices[:selectCount]]

    @staticmethod
    def random(population: List, fitnessVals: List, selectCount: int) -> List:
        """
        This method randomly selects individuals from the population.
        """
        if len(population) != len(fitnessVals):
            raise ValueError("Population and fitness values must be of the same length.")
        if selectCount > len(population):
            raise ValueError("The number of individuals to be selected cannot be greater than the population size.")
        return random.sample(population, selectCount)


class Problem(ABC):
    """
    Abstract class representing an optimization problem.
    """

    @abstractmethod
    def initialize_population(self, population_size: int) -> List:
        """ Initialize the population of the problem. """
        pass

    @abstractmethod
    def fitness(self, genome) -> float:
        """ Calculate the fitness of the genome based on image difference. """
        pass

    @abstractmethod
    def mutate(self, individual, mutation_rate: float) -> Tuple:
        """ Mutate an individual with a given mutation rate. """
        pass

    @abstractmethod
    def crossover(self, parent1, parent2) -> Tuple:
        """ Perform crossover between two parents. """
        pass


class PolygonImageProblem(Problem):
    """
    Class representing the optimization problem for generating images using polygons.
    """

    def __init__(self, target_image_path: str, num_polygons: int, num_vertices: int, mutation_rate: float):
        self.target_image = Image.open(target_image_path).convert("RGB")
        self.num_polygons = num_polygons
        self.num_vertices = num_vertices
        self.mutation_rate = mutation_rate

    def initialize_population(self, population_size: int) -> List[dict]:
        """
        Initialize the population of polygons.
        """
        population = []
        for _ in range(population_size):
            genome = []
            for _ in range(self.num_polygons):
                polygon = {
                    'vertices': [(random.randint(0, self.target_image.width), random.randint(0, self.target_image.height)) 
                                 for _ in range(self.num_vertices)],
                    'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                }
                genome.append(polygon)
            population.append(genome)
        return population

    def fitness(self, genome) -> float:
        """
        Calculate the fitness of the genome based on image difference.
        """
        image = self.draw_image(genome)
        diff = np.array(image).astype(np.float32) - np.array(self.target_image).astype(np.float32)
        return -np.mean(diff)  # We want to maximize similarity, so using negative difference

    def mutate(self, individual, mutation_rate: float) -> Tuple:
        """
        Mutate an individual with a given mutation rate.
        """
        mutated_individual = []
        for polygon in individual:
            if random.random() < mutation_rate:
                # Mutate vertices
                mutated_polygon = {
                    'vertices': [(self.mutate_vertex(vertex[0], self.target_image.width),
                                  self.mutate_vertex(vertex[1], self.target_image.height)) 
                                 for vertex in polygon['vertices']],
                    'color': self.mutate_color(polygon['color'])
                }
                mutated_individual.append(mutated_polygon)
            else:
                mutated_individual.append(polygon)
        return mutated_individual,

    def crossover(self, parent1, parent2) -> Tuple:
        """
        Perform crossover between two parents.
        """
        crossover_point = random.randint(1, self.num_polygons - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate_vertex(self, vertex, limit):
        """
        Mutate a vertex within a limit.
        """
        return max(0, min(vertex + random.randint(-5, 5), limit - 1))

    def mutate_color(self, color):
        """
        Mutate a color value.
        """
        return tuple(max(0, min(channel + random.randint(-20, 20), 255)) for channel in color)

    def draw_image(self, genome) -> Image:
        """
        Draw an image using polygons in the genome.
        """
        image = Image.new("RGB", self.target_image.size)
        draw = ImageDraw.Draw(image)
        for polygon in genome:
            draw.polygon(polygon['vertices'], fill=polygon['color'])
        return image


class EvolutionaryAlgorithm:
    """
    Class representing an evolutionary algorithm for optimization problems.
    """

    def __init__(self, problem: Problem, parent_selection_method: str, survival_selection_method: str):
        self.problem = problem
        self.population = []
        self.parent_selection_method = parent_selection_method
        self.survival_selection_method = survival_selection_method
        self.best_individual = None
        self.best_fitness = float('-inf')  # Initialize with negative infinity for maximization problems

    def initialize_population(self, population_size: int):
        """
        Initialize the population for the evolutionary algorithm.
        """
        self.population = self.problem.initialize_population(population_size)
        self.update_best_individual()  # Initialize the best individual

    def run(self, num_generations: int):
        """
        Run the evolutionary algorithm for a given number of generations.
        """
        for generation in range(num_generations):
            print(f"Generation {generation + 1}/{num_generations}")

            # Parent selection
            self.parent_selection()

            # Mutation and crossover
            offspring = self.generate_offspring()

            # Survival selection
            self.survival_selection(offspring)

            # Update best individual
            self.update_best_individual()

            # Print information
            best_fitness = self.problem.fitness(self.best_individual)
            average_fitness = sum(self.problem.fitness(individual) for individual in self.population) / len(self.population)
            print(f"Best individual in generation {generation + 1},", f"Fitness: {abs(best_fitness)}")
            print(f"Average Fitness: {abs(average_fitness)}")

            # Save the updated image after each generation (if applicable)
            # self.save_image(generation)

    def parent_selection(self):
        """
        Perform parent selection based on the specified method.
        """
        fitness_values = [self.problem.fitness(individual) for individual in self.population]
        if self.parent_selection_method == "fitness_proportional":
            self.population = SelectionSchemes.fitnessProportional(self.population, fitness_values, len(self.population))
        elif self.parent_selection_method == "rank_based":
            self.population = SelectionSchemes.rankBased(self.population, fitness_values, len(self.population))
        elif self.parent_selection_method == "binary_tournament":
            self.population = SelectionSchemes.binaryTournament(self.population, fitness_values, len(self.population))
        elif self.parent_selection_method == "truncation":
            self.population = SelectionSchemes.truncation(self.population, fitness_values, len(self.population))
        elif self.parent_selection_method == "random":
            self.population = SelectionSchemes.random(self.population, fitness_values, len(self.population))
        else:
            raise ValueError("Invalid parent selection method specified.")

    def generate_offspring(self):
        """
        Generate offspring through mutation and crossover.
        """
        offspring = []
        while len(offspring) < len(self.population):
            parent1, parent2 = random.choices(self.population, k=2)
            child1, child2 = self.problem.crossover(parent1, parent2)
            child1 = self.problem.mutate(child1, self.problem.mutation_rate)[0]
            child2 = self.problem.mutate(child2, self.problem.mutation_rate)[0]
            offspring.extend([child1, child2])
        return offspring

    def survival_selection(self, offspring):
        """
        Perform survival selection based on the specified method.
        """
        fitness_values = [self.problem.fitness(individual) for individual in self.population]
        combined_population = self.population + offspring
        combined_fitness_values = fitness_values + [self.problem.fitness(individual) for individual in offspring]
        if self.survival_selection_method == "truncation":
            self.population = SelectionSchemes.truncation(combined_population, combined_fitness_values, len(self.population))
        elif self.survival_selection_method == "random":
            self.population = SelectionSchemes.random(combined_population, combined_fitness_values, len(self.population))
        elif self.survival_selection_method == "rank_based":
            self.population = SelectionSchemes.rankBased(combined_population, combined_fitness_values, len(self.population))
        elif self.survival_selection_method == "binary_tournament":
            self.population = SelectionSchemes.binaryTournament(combined_population, combined_fitness_values, len(self.population))
        elif self.survival_selection_method == "fitness_proportional":
            self.population = SelectionSchemes.fitnessProportional(combined_population, combined_fitness_values, len(self.population))
        else:
            raise ValueError("Invalid survival selection method specified.")

    def update_best_individual(self):
        """
        Update the best individual found so far.
        """
        for individual in self.population:
            fitness = self.problem.fitness(individual)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual

if __name__ == "__main__":
    target_image_path = "monalisa.jpg"
    num_polygons = 50
    num_vertices = 5
    mutation_rate = 0.05
    population_size = 200
    num_generations = 1000

    problem = PolygonImageProblem(target_image_path, num_polygons, num_vertices, mutation_rate)
    evolutionary_algorithm = EvolutionaryAlgorithm(problem, "fitness_proportional", "fitness_proportional")
    evolutionary_algorithm.initialize_population(population_size)
    evolutionary_algorithm.run(num_generations)