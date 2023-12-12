from collections import defaultdict
from optimization import load_csv_and_filter, evaluate_fitness
from libraries import log
import random
import sys
import json
import multiprocessing
import random
import numpy as np
import math

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define the objective function for feature selection
def objective_function(features):
    return sum(features)

# Initialize the population of solutions (bees)
def initialize_population(pop_size, num_features):
    return [[random.choice([0, 1]) for _ in range(num_features)] for _ in range(pop_size)]

# Select employed bees to explore new solutions
def employed_bees_phase(population, fitness_scores, max_trials):
    new_population = []
    for i, solution in enumerate(population):
        trial_count = 0
        while trial_count < max_trials:
            neighbor_index = random.randint(0, len(population) - 1)
            if neighbor_index == i:
                continue
            j = random.randint(0, len(solution) - 1)
            new_solution = solution.copy()
            new_solution[j] = 1 - new_solution[j]
            if objective_function(new_solution) > fitness_scores[i]:
                solution = new_solution
            trial_count += 1
        new_population.append(solution)
    return new_population

def onlooker_bees_phase(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fit / total_fitness for fit in fitness_scores]
    pop_size = len(population)
    new_population = population.copy()  # Create a copy of the population

    for _ in range(pop_size):
        selected_bee_index = np.random.choice(pop_size, p=probabilities)
        selected_bee = new_population[selected_bee_index]
        j = random.randint(0, len(selected_bee) - 1)
        selected_bee[j] = 1 - selected_bee[j] # Flip the feature
        if objective_function(selected_bee) <= fitness_scores[selected_bee_index]: # If new solution is worse, revert the change
            selected_bee[j] = 1 - selected_bee[j]

    return new_population

# Select scout bees to replace abandoned solutions
def scout_bees_phase(population, max_trials):
    return [[random.choice([0, 1]) for _ in range(len(solution))] if random.random() < 1 / (1 + max_trials) else solution for solution in population]

# ABC feature selection algorithm
def abc_feature_selection(population_size, solution_size, max_trials, num_cores, log_file_path, classes_file_path, train_file_paths, num_of_packets_to_process, fields_file_path, classifier_index, weights, num_of_iterations, max_num_of_generations, classifiers):
    pre_solutions = defaultdict(float)

    # Load classes
    try:
        with open(classes_file_path, 'r') as file:
            classes = json.loads(file.readline())
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {classes_file_path} does not exist.")

    # Load the packets
    log("loading packets...", log_file_path)
    packets_1 = []
    packets_2 = []

    # Read header from fields file
    try:
        with open(fields_file_path, 'r') as file:
            header = [line.strip() for line in file.readlines()] + ['label']
            packets_1.append(header)
            packets_2.append(header)
    except FileNotFoundError:
        print(f"The file {fields_file_path} does not exist.")
        sys.exit(1)

    packets_1.extend(element for element in load_csv_and_filter(classes, train_file_paths[0], num_of_packets_to_process, log_file_path))
    packets_2.extend(element for element in load_csv_and_filter(classes, train_file_paths[1], num_of_packets_to_process, log_file_path))

    log("", log_file_path)

    population = initialize_population(population_size, solution_size)
    best_solution = None
    best_fitness = -math.inf

    fitness_scores = [-math.inf for _ in range(population_size)]

    consecutive_same_solution_count = 1
    generation = 0

    while consecutive_same_solution_count < num_of_iterations and generation < max_num_of_generations:
        if best_solution is not None:
            employed_population = employed_bees_phase(population, fitness_scores, max_trials)
            onlooker_population = onlooker_bees_phase(employed_population, fitness_scores)
            population = scout_bees_phase(onlooker_population, max_trials)
            population[-1] = best_solution # Preserve the best solution from the previous generation

        # Evaluate the fitness of each solution in the population using multi-threading
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions, weights, classifiers) for solution in population])

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

def run(train_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations, fields_file_path, num_cores, classifiers):
    # Configuration parameters
    population_size = 50
    max_trials = 5

    # Determine solution size (number of features)
    try:
        with open(train_file_paths[0], 'r') as file:
            solution_size = len(file.readline().split(',')) - 1
    except FileNotFoundError:
        print(f"The file {train_file_paths[0]} does not exist.")
        sys.exit(1)

    return abc_feature_selection(
        population_size, solution_size, max_trials, num_cores, log_file_path, classes_file_path,
        train_file_paths, num_of_packets_to_process, fields_file_path, classifier_index, weights,
        num_of_iterations, max_num_of_generations, classifiers
    )
