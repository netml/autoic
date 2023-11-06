from collections import defaultdict
from libraries import log
import numpy as np
import ml
import csv
import random

def evaluate_fitness(solution, packets_1, packets_2, classifier_index, pre_solutions, weights, classifiers):
    pre_solutions_gen = defaultdict(float)
    n = len(solution)
    k = sum(solution)

    # If no features are to be selected
    if k == 0:
        return 0.0, pre_solutions_gen
    elif n == 1:
        return 1.0, pre_solutions_gen

    key = ''.join(map(str, solution))

    # Acquire the lock before reading pre_solutions
    if key in pre_solutions:
        return pre_solutions[key], pre_solutions_gen

    # Append 1 to the end so that it doesn't filter out the 'class' column
    solution_new = solution + [1]

    # Filter features
    filtered_packets_1 = [[col for col, m in zip(row, solution_new) if m] for row in packets_1]
    filtered_packets_2 = [[col for col, m in zip(row, solution_new) if m] for row in packets_2]
    
    fitness_1 = ml.classify(filtered_packets_1, filtered_packets_2, classifier_index, classifiers)[0]
    fitness_2 = ml.classify(filtered_packets_2, filtered_packets_1, classifier_index, classifiers)[0]
    average_accuracy = np.mean([fitness_1, fitness_2])

    # Calculate feature accuracy
    feature_accuracy = (n - k) / (n - 1)

    # Calculate fitness as a weighted combination of average accuracy and feature accuracy
    fitness = weights[0] * average_accuracy + weights[1] * feature_accuracy

    # Acquire the lock before updating pre_solutions
    pre_solutions_gen[key] = fitness

    return fitness, pre_solutions_gen

def load_csv_and_filter(classes, fitness_function_file_path, n, log_file_path):
    packets = []
    for i in range(len(classes)):
        log("reading from " + classes[str(i)] + "...", log_file_path)
        with open(fitness_function_file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None) # Skip the header row

            lines = [row for row in csv_reader if row[-1] == str(i)]
            random.shuffle(lines)

            no_of_packets_to_keep = len(lines) if n == 0 else min(n, len(lines))
            packets.extend(lines[:no_of_packets_to_keep]) # Append the selected lines directly to packets

    packets = [[float(value) for value in packet] for packet in packets]
    
    return packets
