from sklearn.metrics import classification_report, f1_score
from libraries import log
import csv
import sys

def load_csv(file_path):
    try:
        with open(file_path, 'r') as f: # Open the CSV file for reading
            reader = csv.reader(f) # Create a CSV reader object
            data = [next(reader)] + [[float(token) for token in row] for row in reader] # read the header as string and the rest as float
        return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        sys.exit(1)

def remove_duplicates_list_list(list_of_lists):
    unique_sublists = set(tuple(sublist) for sublist in list_of_lists[1:]) # Convert all sublists except the first one to tuples (which are hashable)
    result = [list_of_lists[0]] + [list(sublist) for sublist in unique_sublists] # Combine the first sublist with the unique sublists back into a result list
    return result # 'result' contains the first occurrence from list_of_lists and unique sublists

def extract_features_and_labels(data, label_column_index):
    features = [row[:label_column_index] + row[label_column_index+1:] for row in data] # Extract features by removing the label column from each row
    labels = [row[label_column_index] for row in data] # Extract labels by selecting the label column from each row
    return features, labels

def train_and_evaluate_classifier(classifier_index, train_features, train_labels, test_features, test_labels, classifiers):
    if classifier_index < 0 or classifier_index >= len(classifiers):
            raise ValueError("Invalid classifier index")

    clf = classifiers[classifier_index][1]
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    f1 = f1_score(test_labels, predictions, average='macro')

    return f1, predictions, test_labels

def classify(train, test, classifier_index, classifiers):
    try:
        label_column_index = train[0].index('label') # Find the index of the label column in the header

        # Extract features and labels for training and testing
        train_features, train_labels = extract_features_and_labels(train[1:], label_column_index)
        test_features, test_labels = extract_features_and_labels(test[1:], label_column_index)

        # Train and evaluate the classifier
        return train_and_evaluate_classifier(classifier_index, train_features, train_labels, test_features, test_labels, classifiers)
    except ValueError as e:
        print(f"Error: {e}")

def classify_after_filtering(solution, fitness_function_file_paths, test_file_path, classifier_index, log_file_path, classifiers, filter):
    # Load training and testing data, remove duplicates (fitness_function_file_paths[1] except for header)
    train = remove_duplicates_list_list(load_csv(fitness_function_file_paths[0]) + load_csv(fitness_function_file_paths[1])[1:])
    test = remove_duplicates_list_list(load_csv(test_file_path))

    if filter: # Apply feature filtering if necessary
        solution_new = list(solution) + [1] # Append 1 to the end so that it doesn't filter out the 'class' column
        train = [[col for col, m in zip(row, solution_new) if m] for row in train]
        test = [[col for col, m in zip(row, solution_new) if m] for row in test]

    f1_score_average, predictions, test_labels = classify(train, test, classifier_index, classifiers) # Classify and evaluate

    # Log results
    log("\nF1-Score: " + str(f1_score_average), log_file_path)
    log("\nClassification Report:", log_file_path)
    log(classification_report(test_labels, predictions, zero_division=0), log_file_path)
