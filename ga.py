from collections import defaultdict
from optimization import load_csv_and_filter, evaluate_fitness
from libraries import log
import numpy as np
import math
import random
import sys
import json
import multiprocessing

# Set a random seed for reproducibility
random.seed(42)

def initialize_population(pop_size, solution_size):
    # Initialize a population
    return [[1 for _ in range(solution_size)] for _ in range(pop_size)]

def select_parents(population, fitness_scores):
    # Implement Elitism: Select parents based on fitness (roulette wheel selection)
    total_fitness = sum(fitness_scores)
    probabilities = [score / total_fitness for score in fitness_scores]
    num_of_elitist_solution = round(len(population) * 0.2)  # Determine the number of elite solutions (20%)
    index_of_max = fitness_scores.index(max(fitness_scores))  # Find the index of the best solution
    # Select parents using roulette wheel selection
    parents = random.choices(population, weights=probabilities, k=len(population) - num_of_elitist_solution)
    # Keep the best solution from the previous generation
    parents.extend([population[index_of_max]] * num_of_elitist_solution)
    return parents

def uniform_crossover(parent1, parent2, crossover_rate):
    # Perform uniform crossover with a given crossover rate
    return [random.choice([bit1, bit2])
            if random.random() <= crossover_rate else bit1 for bit1, bit2 in zip(parent1, parent2)]

def mutate(solution, mutation_rate):
    # Apply bit-flip mutation with a given mutation rate
    return [bit if random.random() >= mutation_rate else 1 - bit for bit in solution]

def genetic_algorithm(pop_size, solution_size, mutation_rate, crossover_rate, optimization_train_file_path, classifier_index,
                      num_of_iterations, classes_file_path, num_of_packets_to_process, weights, log_file_path,
                      max_num_of_generations, fields_file_path, num_cores, classifiers):

    pre_solutions = defaultdict(float)
    
    # Load classes
    try:
        with open(classes_file_path, 'r') as file:
            classes = json.loads(file.readline())
    except FileNotFoundError:
        print(f"The file {classes_file_path} does not exist.")
        sys.exit(1)

    # Load the packets
    log("loading packets...", log_file_path)

    # Assuming packets_1 and packets_2 are already defined and contain existing data
    packets_1, packets_2 = load_csv_and_filter(classes, optimization_train_file_path, num_of_packets_to_process,
                                                       log_file_path, fields_file_path)

    log("", log_file_path)

    fitness_scores = []
    population = initialize_population(pop_size, solution_size)
    best_solution = None
    best_fitness = -math.inf

    consecutive_same_solution_count = 1
    generation = 0

    while consecutive_same_solution_count < num_of_iterations and generation < max_num_of_generations:
        if best_solution is not None:
            parents = select_parents(population, fitness_scores)  # Select parents for reproduction using Elitism
            new_population = []  # Create a new population through crossover and mutation

            while len(new_population) < pop_size:
                parent1, parent2 = random.choices(parents, k=2)
                child = uniform_crossover(parent1, parent2, crossover_rate)
                child = mutate(child, mutation_rate)
                new_population.append(child)

            population = new_population # Replace the old population with the new population
        
        # Evaluate the fitness of each solution in the population using multi-threading
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions,
                                                       weights, classifiers) for solution in population])

        pool.close()
        pool.join()

        fitness_scores = []
        for result in results:
            fitness_scores.append(result[0])
            pre_solutions.update(result[1])

        # Find the best solution in the current generation
        generation_best_index = np.argmax(fitness_scores)
        generation_best_fitness = fitness_scores[generation_best_index]
        generation_best_solution = population[generation_best_index]

        # Track and display the best solution in this generation            
        if generation_best_fitness > best_fitness:
            best_solution = generation_best_solution
            best_fitness = generation_best_fitness
            consecutive_same_solution_count = 1
        else:
            consecutive_same_solution_count += 1

        sol_str = ''.join(map(str, best_solution))
        log(f"Generation {generation + 1}:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}", log_file_path)
        
        generation += 1

    log("", log_file_path)

    return best_solution, best_fitness

def run(optimization_train_file_path, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights,
        log_file_path, max_num_of_generations, fields_file_path, num_cores, classifiers):
    # Configuration parameters
    population_size = 50
    mutation_rate = 0.015
    crossover_rate = 0.5

    # Determine solution size (number of features)
    try:
        with open(optimization_train_file_path, 'r') as file:
            solution_size = len(file.readline().split(',')) - 1
    except FileNotFoundError:
        print(f"The file {optimization_train_file_path} does not exist.")
        sys.exit(1)

    return genetic_algorithm(
        population_size, solution_size, mutation_rate, crossover_rate, optimization_train_file_path,
        classifier_index, num_of_iterations, classes_file_path, num_of_packets_to_process,
        weights, log_file_path, max_num_of_generations, fields_file_path, num_cores, classifiers
    )
