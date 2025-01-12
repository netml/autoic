# verify the validation checks after the user providing the features
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from libraries import log
from sklearn.naive_bayes import GaussianNB
import multiprocessing
import sys
import os
import ga
import ml
import aco
import bee
import report
import extract
import libraries

if __name__ == '__main__':
    # Exit if no command-line arguments are given
    if len(sys.argv) < 2:
        print("Usage: python script.py arg1 arg2...")
        sys.exit(1)

    # Parameters
    classifier_index = ""
    max_num_of_generations = 100
    num_of_iterations = 10
    num_of_packets_to_process = 0
    num_of_batches = 3
    weights = [0.9, 0.1]
    mode = ""
    folder = ""
    protocol = ""
    tshark_filter = ""
    run_number = 1
    num_cores = multiprocessing.cpu_count() - 1  # Number of cores to use
    statistical_features_on = False
    shap_features_on = False
    blacklist_check = True
    pcap_file_names = None
    pcap_file_paths = None
    shap_fold_size = 10
    classifiers = [
        ("DT", DecisionTreeClassifier(random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("SVC", SVC(random_state=42)),
        ("LiSVC", LinearSVC(random_state=42, dual='auto', C=1.0, max_iter=10000)),
        ("MLP", MLPClassifier(
            hidden_layer_sizes=(512, 256), # Reflecting the two dense layers in your TensorFlow model
            max_iter=500,                  # Equivalent to the 500 epochs
            alpha=0.001,                   # Regularization strength (kernel_regularizer in TensorFlow)
            learning_rate='adaptive',      # To dynamically adjust the learning rate
            learning_rate_init=0.001,      # Initial learning rate matching RMSprop's learning rate
            batch_size=32,                 # Matches the batch size in TensorFlow
            early_stopping=True,           # To stop training based on validation loss
            validation_fraction=0.3,       # Matches the 30% validation split
            n_iter_no_change=30,           # Reflects patience for early stopping
            activation='relu',             # Matches the activation functions in TensorFlow
            solver='adam',                 # Closest Scikit-learn equivalent to RMSprop optimizer
            random_state=42                # Ensures reproducibility
        )),
        ("GNB", GaussianNB()),
        ("KNN", KNeighborsClassifier()),
        ("LR", LogisticRegression(solver='saga', multi_class='ovr', max_iter=10000))
    ]

    # Loop through command-line arguments starting from the second element
    index = 1
    while index < len(sys.argv):
        if sys.argv[index] in ('-p', '--protocol'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -p/--protocol option")
                sys.exit(1)
            protocol = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-t', '--tshark-filter'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -t/--tshark-filter option")
                sys.exit(1)
            tshark_filter = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-i', '--iteration'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -i/--iteration option")
                sys.exit(1)
            num_of_iterations = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-g', '--generation'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -g/--generation option")
                sys.exit(1)
            max_num_of_generations = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-w', '--weights'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -w/--weights option")
                sys.exit(1)
            weights = [float(value) for value in sys.argv[index + 1].split(',')]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in '-n':
            if index + 1 >= len(sys.argv):
                print("Missing value for -n option")
                sys.exit(1)
            num_of_packets_to_process = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-nc', 'num-cores'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -nc/--num-cores option")
                sys.exit(1)
            num_cores = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-f', '--folder'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -f/--folder option")
                sys.exit(1)
            folder = libraries.fix_trailing_character(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-r', '--run-number'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -r/--run-number option")
                sys.exit(1)
            run_number = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-s', '--statistics'):
            statistical_features_on = True
            index += 1
        elif sys.argv[index] in ('-sf', '--shap-fold-size'):
            shap_fold_size = int(sys.argv[index + 1])
            index += 1
        elif sys.argv[index] in '--shap':
            shap_features_on = True
            index += 1
        elif sys.argv[index] in ('-m', '--mode'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -m/--mode option")
                sys.exit(1)
            mode = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-h', '--help'):
            libraries.print_usage()
            sys.exit(0)
        elif sys.argv[index] in ('-c', '--classifier'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -c/--classifier option")
                sys.exit(1)
            classifier_index = sys.argv[index+1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-nb', '--no-blacklist'):
            blacklist_check = False
            index += 1
        elif sys.argv[index] in ('-b', '--num-of-batches'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -b/--num-of-batches option")
                sys.exit(1)
            num_of_batches = int(sys.argv[index + 1])
            index += 2
        else:
            print(f"Unknown parameter! '{sys.argv[index]}'")
            sys.exit(1)

    # Validation checks
    required_params = {
        "folder": folder,
        "protocol": protocol,
        "mode": mode,
    }

    # Include classifier_index only if the mode requires it
    if mode in ['ga', 'aco', 'abc']:
        required_params["classifier_index"] = classifier_index

    for param, value in required_params.items():
        if not value:
            print(f"Missing required parameter: {param}")
            sys.exit(1)

    # Set parameters
    pcap_folder_path = folder + "pcap"
    classes_file_path = f'{folder}/{protocol}/classes.json'
    extracted_field_list_file_path = f'{folder}/{protocol}/original_dataset_fields.txt'
    shap_extracted_field_list_file_path = f'{folder}/{protocol}/shap_dataset_fields.txt'
    split_file_paths = [f'{folder}{protocol}/original_dataset_split_{i+1}.csv' for i in range(num_of_batches)]
    batch_file_paths = [[f'{folder}{protocol}/original_dataset_batch_{i+1}_train.csv' for i in range(num_of_batches)],
                        [f'{folder}{protocol}/original_dataset_batch_{i+1}_test.csv' for i in range(num_of_batches)]]
    shap_file_paths = [f'{folder}{protocol}/shap_dataset_batch_{i+1}.csv' for i in range(num_of_batches)]
    filters_folder = os.path.join(os.path.dirname(__file__), "filters")
    blacklist_file_path = f'{filters_folder}/blacklist.txt'
    feature_names_file_path = f'{filters_folder}/{protocol}.txt'
    protocol_folder_path = f'{folder}{protocol}'
    if mode == 'extract':
        pcap_file_names = sorted([f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')])
        pcap_file_paths = [folder + "pcap/" + file_name for file_name in pcap_file_names]
    order_of_batches = libraries.generate_specific_combinations(num_of_batches)

    # Validation checks
    libraries.check_path_exists(folder, 'Workspace folder')
    libraries.check_path_exists(filters_folder, 'filters folder', is_folder=True)
    if mode == 'extract':
        libraries.check_path_exists(pcap_folder_path, 'pcap folder')
        if len(pcap_file_names) == 0:
            print("There are no pcap files in the 'pcap' folder.")
            sys.exit(1)
    elif mode in ['ga', 'aco', 'abc']:
        if classifier_index == "":
            print("Classifier index not given!")
            sys.exit(1)

    # Run the mode
    if mode == 'extract':
        extract.run(
            blacklist_check, blacklist_file_path, feature_names_file_path, protocol_folder_path, split_file_paths,
            shap_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, extracted_field_list_file_path,
            shap_extracted_field_list_file_path, statistical_features_on, tshark_filter,
            f'{folder}{protocol}/original_dataset.csv',
            f'{folder}{protocol}/shap_dataset.csv', shap_fold_size, shap_features_on, batch_file_paths,
            num_of_batches
        )
        print("done...")
    elif mode == 'ga' or mode == 'aco' or mode == 'abc':
        for batch_number, order_of_batch in enumerate(order_of_batches):
            log_file_path = (
                folder + "/" + protocol + "/" +
                ("shap" if shap_features_on else "original") + "_dataset_results" +
                "_num_" + str(num_of_packets_to_process) +
                "_mode_" + str(mode) +
                "_clf_" + classifier_index +
                "_batch_" + str(batch_number + 1) +
                "_run_" + str(run_number) +
                ".txt"
            )

            if os.path.exists(log_file_path):
                os.remove(log_file_path)  # Remove previous log file

            dataset_type = "shap" if shap_features_on else "original"

            optimization_train_file_path = batch_file_paths[0][batch_number]
            optimization_test_file_path = batch_file_paths[1][batch_number]

            best_solution = None
            best_fitness = 0

            if mode == 'ga':
                log("running GA...\n", log_file_path)
                best_solution, best_fitness = ga.run(
                    optimization_train_file_path, int(classifier_index), classes_file_path,
                    num_of_packets_to_process, num_of_iterations, weights,
                    log_file_path, max_num_of_generations,
                    (shap_extracted_field_list_file_path if shap_features_on else extracted_field_list_file_path),
                    num_cores, classifiers
                )
            elif mode == 'aco':
                log("running ACO...\n", log_file_path)
                best_solution, best_fitness = aco.run(
                    optimization_train_file_path, int(classifier_index), classes_file_path,
                    num_of_packets_to_process, num_of_iterations, weights,
                    log_file_path, max_num_of_generations,
                    (shap_extracted_field_list_file_path if shap_features_on else extracted_field_list_file_path),
                    num_cores, classifiers
                )
            elif mode == 'abc':
                log("running ABC...\n", log_file_path)
                best_solution, best_fitness = bee.run(
                    optimization_train_file_path, int(classifier_index), classes_file_path,
                    num_of_packets_to_process, num_of_iterations, weights,
                    log_file_path, max_num_of_generations,
                    (shap_extracted_field_list_file_path if shap_features_on else extracted_field_list_file_path),
                    num_cores, classifiers
                )

            # Read the extracted field list
            extracted_field_list = []
            file_path = shap_extracted_field_list_file_path if shap_features_on else extracted_field_list_file_path
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    extracted_field_list = [line.strip() for line in file.readlines()]

            # Print best solution and the features selected
            sol_str = ''.join(map(str, best_solution))
            log(f"Best Solution:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}",
                log_file_path)

            # Get the selected features
            selected_field_list = []
            for i in range(len(best_solution)):
                if best_solution[i] == 1:
                    selected_field_list.append(extracted_field_list[i])

            selected_features_batch_paths = [
                f'{folder}{protocol}/{dataset_type}_{mode}_dataset_batch_{batch_number+1}_train.csv',
                f'{folder}{protocol}/{dataset_type}_{mode}_dataset_batch_{batch_number+1}_test.csv'
            ]

            # Create the selected features CSV files
            columns_to_keep = selected_field_list + ['label']
            libraries.filter_columns(
                f'{folder}{protocol}/{dataset_type}_dataset_batch_{order_of_batch[batch_number]}_train.csv',
                selected_features_batch_paths[0], columns_to_keep
            )
            libraries.filter_columns(
                f'{folder}{protocol}/{dataset_type}_dataset_batch_{order_of_batch[batch_number]}_test.csv',
                selected_features_batch_paths[1], columns_to_keep
            )

            # Print the selected features
            log("\nSelected features:", log_file_path)
            for i in range(len(selected_field_list)):
                log(selected_field_list[i], log_file_path)

            # Print the classification result on test data using selected features
            log("", log_file_path)
            log("Selected feature-set results:", log_file_path)
            ml.classify_after_filtering(best_solution, optimization_train_file_path, optimization_test_file_path,
                                        int(classifier_index), log_file_path, classifiers, True)
            
            # Print the classification result on test data using all features
            log("All feature-set results:", log_file_path)
            ml.classify_after_filtering(best_solution, optimization_train_file_path, optimization_test_file_path,
                                        int(classifier_index), log_file_path, classifiers, False)
    elif mode == 'report':
        report.run(folder + protocol, classifiers, classifier_index)
    else:
        print("Unknown entry for the mode!")
