from collections import defaultdict
import math
from optimization import load_csv_and_filter, evaluate_fitness
from libraries import log
import random
import threading
import json
import multiprocessing
import sys

# Set a random seed for reproducibility
random.seed(42)

# Define a lock for synchronization
thread_lock = threading.Lock()

# Define the ACO algorithm
def ant_colony_optimization(num_of_ants, num_of_iterations, pheromone_decay, pheromone_strength, train_file_paths, classifier_index, solution_size, classes_file_path, num_of_packets_to_process, weights, log_file_path, max_num_of_generations, fields_file_path, num_cores, classifiers):
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

    packets_1.extend(element for element in load_csv_and_filter(classes, train_file_paths[0], num_of_packets_to_process, log_file_path))
    packets_2.extend(element for element in load_csv_and_filter(classes, train_file_paths[1], num_of_packets_to_process, log_file_path))

    log("", log_file_path)

    # Initialize the pheromone matrix with equal values for each ant
    pheromones = [1.0] * num_of_ants

    # Define the ant behavior
    def ant_behavior(ant_index, fitness_values):
        with thread_lock:
            pheromones[ant_index] += pheromone_strength * fitness_values[ant_index]  # Increase pheromone based on fitness
            pheromones[ant_index] *= pheromone_decay  # Decay pheromone

    # Run the algorithm until the same best solution is produced 'n_iterations' times in a row
    best_solution_counter = 1
    iteration_counter = 0
    best_solution = None
    best_fitness = -math.inf

    while best_solution_counter < num_of_iterations and iteration_counter < max_num_of_generations:
        solutions = [[random.randint(0, 1) for _ in range(solution_size)] for _ in range(num_of_ants)]

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.starmap(evaluate_fitness, [(solution, packets_1, packets_2, classifier_index, pre_solutions, weights, classifiers) for solution in solutions])

        pool.close()
        pool.join()

        fitness_values = []
        for result in results:
            fitness_values.append(result[0])
            pre_solutions.update(result[1])

        current_best_solution = None
        current_best_fitness = None

        for j in range(num_of_ants):
            ant_behavior(j, fitness_values)

        # Choose the best solution based on the fitness values
        ant_fitness_max_index = fitness_values.index(max(fitness_values))
        current_best_solution = solutions[ant_fitness_max_index]
        current_best_fitness = evaluate_fitness(current_best_solution, packets_1, packets_2, classifier_index, pre_solutions, weights, classifiers)[0]

        # If the current best solution is better than the previous best solution, update the best solution and best fitness
        if current_best_fitness > best_fitness:
            best_solution = current_best_solution
            best_fitness = current_best_fitness
            best_solution_counter = 1
        else:
            best_solution_counter += 1

        # Print current best solution with a grid of filled squares for 1 and empty squares for 0
        sol_str = ''.join(map(str, best_solution))
        log(f"Generation {iteration_counter + 1}:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}", log_file_path)

        iteration_counter += 1

    log("", log_file_path)

    # Return the best solution and its fitness value
    return (best_solution, best_fitness)

def run(train_file_paths, classifier_index, classes_file_path, num_of_packets_to_process, num_of_iterations, weights, log_file_path, max_num_of_generations, fields_file_path, num_cores, classifiers):
    # Configuration parameters
    num_of_ants = 10
    pheromone_strength = 1
    pheromone_decay = 0.5

    # Determine solution size (number of features)
    try:
        with open(train_file_paths[0], 'r') as file:
            solution_size = len(file.readline().split(',')) - 1
    except FileNotFoundError:
        print(f"The file {train_file_paths[0]} does not exist.")

    return ant_colony_optimization(
        num_of_ants, num_of_iterations, pheromone_decay, pheromone_strength, train_file_paths,
        classifier_index, solution_size, classes_file_path, num_of_packets_to_process, weights,
        log_file_path, max_num_of_generations, fields_file_path, num_cores, classifiers
    )
