from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from libraries import log
from wittgenstein import RIPPER
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
import multiprocessing
import pandas as pd
import subprocess
import csv
import sys
import os
import re
import ga
import ml, aco, bee
import json
import statistics
import report

def print_usage():
    print("""
    Usage: python3 autoic [OPTIONS]

    Options:

    -p, --protocol <protocol>       Specify the protocol to use. This option is required.
                                    e.g. 'http'

    -t, --tshark-filter <filter>    Apply a tshark filter to the captured traffic. The filter
                                    should be a valid tshark filter expression. This option
                                    is used to filter the captured traffic. This option is
                                    optional. Similar to '-Y' option in tshark. e.g. 'ip'
                                    which is equivalent to 'tshark -Y ip'.

    -b, --batch <batches>           Define the order of processing batches. This should be a 
                                    comma-separated list of integers, where the last integer
                                    indicates the test batch. e.g. '1,2,3' means that the
                                    first batch is the training batch, the second batch is
                                    the validation batch, and the third batch is the test
                                    batch. This option is required.

    -i, --iteration <number>        Set the number of iterations for the process. The value
                                    should be an integer. This option is set to 10 by default.

    -g, --generation <number>       Specify the maximum number of generations. The value should 
                                    be an integer. This option is set to 100 by default.

    -w, --weights <weights>         Set the weights for the tool's calculations. This should be 
                                    two comma-separated list of floating-point numbers, where
                                    the first number is the weight for the classification and
                                    the second number is the weight for the number of features.
                                    This option is set to '0.9,0.1' by default.

    -n <number>                     Set the number of packets to process. The value should be an 
                                    integer.

    -nc, --num-cores <number>       Specify the maximum number of CPU cores to use. The value
                                    should be an integer. This option is by default set to '0'
                                    which means that the tool will use all available CPU cores
                                    minus 1 by default. One core is reserved for Operating
                                    System processes.

    -f, --folder <path>             Set the path to the folder where the tool will operate. The 
                                    path should be a valid folder path on the system. Make sure
                                    that it contains a 'pcap' folder with the pcap files.

    -r, --run-number <number>       Define the run number for this instance of the tool. This
                                    parameter is used to distinguish between multiple runs of
                                    the tool. It allows the tool to generate multiple log files
                                    one for each run. If not specified, the tool will not append
                                    the run number to the log file name. Therefore, this parameter
                                    is optional. The value should be an integer.

    -l, --log <path>                Set the path to the log file where the tool will write its 
                                    logs. If not specified, the tool will create a log file
                                    with a default name. The path should be a valid file path.

    -s, --statistics                Enable statistical feature calculation. This option does not 
                                    require a value. If specified, the tool will calculate
                                    statistical features for each numeric feature such as
                                    minimum, maximum, mean, standard deviation, and mode.
                                    This option is disabled by default.

    -m, --mode <mode>               Set the mode for the tool. The mode should be a string 
                                    indicating the operation mode. Possible values are:
                                    'extract', 'report', 'ga', 'aco', and 'abc'. 'Extract'
                                    mode extracts features from pcap files and writes them
                                    to CSV files. 'Report' mode generates a report for the
                                    given protocol. 'GA', 'ACO', and 'ABC' modes run the
                                    respective algorithms. This option is required.

    -c, --classifier <index>        Specify the classifier index to use. The value should be a 
                                    valid index integer. This option is required. Possible
                                    values are:
                                    0: Decision Tree
                                    1: Random Forest
                                    2: SVM
                                    3: Linear SVM
                                    4: MLP
                                    5: Naive Bayes
                                    6: RIPPER
                                    7: KNN
                                    8: Logistic Regression
                                    9: Naive Bayes
                                    
    -h, --help                      Display this help message and exit.

    Note: Replace <placeholders> with actual values without the angle brackets.
    """)

def is_numeric(token):
    try:
        float(token)
        return True
    except ValueError:
        return False

def is_hexadecimal(s):
    return bool(re.match(r"^[0-9A-Fa-f]+$", s))

def calculate_list_average(lst):
    return sum(lst) / len(lst) if lst else 0  # Return the average or 0 if the list is empty

def fix_trailing_character(input_string):
    return input_string.rstrip('/') + '/'  # Remove trailing '/' and add it back

def remove_symbols(s):
    return re.sub(r'[^a-zA-Z0-9]', '', s)

def modify_dataset(fields):
    for j, cell in enumerate(fields):
        if cell:
            tokens = cell.split(',')
            total = sum(convert_token(token) for token in tokens)
            fields[j] = str(total % 0xFFFFFFFF)
        else:
            fields[j] = '-1'
    return fields

def convert_token(token):
    token = token.strip()

    if re.match(r'^0x[0-9a-fA-F]+$', token):
        return int(token, 16)
    elif not is_numeric(token):
        return hash(token)
    else:
        return float(token)

def write_line_to_csv(csv_file_paths, fields, class_counter):
    with open(csv_file_paths[class_counter % len(csv_file_paths)], 'a') as file:
        file.write(','.join(fields) + '\n')

def read_blacklisted_features(blacklist_file_path):
    try:
        with open(blacklist_file_path, 'r') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"The file '{blacklist_file_path}' was not found.")
        sys.exit(1)

def read_and_filter_feature_names(feature_names_file_path, blacklisted_features):
    try:
        with open(feature_names_file_path, 'r') as f:
            feature_names = [feature for feature in f.read().splitlines() if feature not in blacklisted_features]
            feature_names.append('label')
        return feature_names
    except FileNotFoundError:
        print(f"The file '{feature_names_file_path}' was not found.")
        sys.exit(1)

def add_stat_features_to_csv_files(csv_file_paths):
    for file_path in csv_file_paths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Create a list to store the new header
        new_header = []

        # Iterate through the existing header and create new columns for statistics
        for field in df.columns[:-1]:  # Exclude the last column (label)
            new_header.extend([field])
        for field in df.columns[:-1]:  # Exclude the last column (label)
            new_header.extend([f"{field}_min", f"{field}_max", f"{field}_mean", f"{field}_std", f"{field}_mode"])
        new_header.append(df.columns[-1])

        # Calculate statistics for each numeric column
        stats = []
        for field in df.columns[:-1]:  # Exclude the last column (label)
            column_values = df[field].tolist()
            stats.extend([min(column_values), max(column_values), statistics.mean(column_values), statistics.stdev(column_values), statistics.mode(column_values)])

        csv_data = []
        csv_data.append(new_header)

        # Iterate through the DataFrame rows and append the list
        for index, row in df.iterrows():
            if index == 0:
                continue  # Skip the header row
            row_list = row.tolist()
            row_list = row_list[:-1] + stats + [int(row_list[-1])]
            csv_data.append(row_list)

        with open(file_path, 'w') as file:
            for inner_list in csv_data:
                line = ','.join(map(str, inner_list))  # Convert inner list to a comma-separated string
                file.write(line + '\n')

def remove_empty_fields_from_csv_files(csv_file_paths):
    # Read all CSV files into a list of DataFrames
    dfs = [pd.read_csv(file_path, low_memory=False) for file_path in csv_file_paths]

    # Find columns with only one unique value across all DataFrames
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns &= set(df.columns)

    columns_to_remove = set()

    # Check if each common column has only one unique value across all DataFrames
    for col in common_columns:
        common_values = set(dfs[0][col].unique())
        for df in dfs[1:]:
            common_values &= set(df[col].unique())
        if len(common_values) == 1:
            # Ensure we're not looking at the last column of each DataFrame
            if col != dfs[0].columns[-1]:  # Check against the last column name, do not remove the label even if it has a single unique value
                columns_to_remove.add(col)

    if columns_to_remove:
        print("removing empty fields...")
        for i in range(len(csv_file_paths)):
            df = pd.read_csv(csv_file_paths[i], low_memory=False)
            df = df.drop(columns=columns_to_remove, errors='ignore')
            df.to_csv(csv_file_paths[i], index=False)

def write_header_to_csv_files(csv_file_paths, feature_names):
    for i in range(len(csv_file_paths)):
        with open(csv_file_paths[i], 'w') as f:
            csv_line = ','.join(feature_names)
            f.write(f"{csv_line}\n")

def write_remaining_field_list_to_file(csv_file_paths, selected_field_list_file_path):
    selected_field_list = []
    with open(csv_file_paths[0], 'r') as file:
        csv_reader = csv.reader(file)
        selected_field_list = next(csv_reader)

    with open(selected_field_list_file_path, 'w') as file:
        file.write(','.join(selected_field_list[:-1]))

def write_packets_to_csv_files(csv_file_paths, num_of_lines_per_file, csv_data):
    for i in range(len(csv_file_paths)):
        start_idx = i * num_of_lines_per_file
        end_idx = start_idx + num_of_lines_per_file if i < 2 else None
        
        file_data = csv_data[start_idx:end_idx]

        with open(csv_file_paths[i], 'a') as f:
            for line in file_data:
                csv_line = ','.join(line)
                f.write(f"{csv_line}\n")

def split_file_into_thirds(filename, csv_file_paths):
    # Count the number of lines in the file
    with open(filename, 'r') as file:
        line_count = sum(1 for _ in file)

    # Calculate the number of lines per split file
    lines_per_file = line_count // 3

    # Open the original file again to split its contents
    with open(filename, 'r') as file:
        for i in range(1, 4):
            with open(csv_file_paths[i-1], 'a') as outfile:
                for _ in range(lines_per_file + (1 if i <= line_count % 3 else 0)):
                    line = file.readline()
                    # Stop if the file ends before expected
                    if not line:
                        break
                    outfile.write(line)

def remove_duplicates_in_place(file_path):
    seen = set()
    unique_lines = []

    # Read unique lines
    with open(file_path, 'r') as file:
        for line in file:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)

    # Overwrite the file with unique lines
    with open(file_path, 'w') as file:
        for line in unique_lines:
            file.write(line)

def extract_features_from_pcap(blacklist_file_path, feature_names_file_path, protocol_folder_path, csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, selected_field_list_file_path, statistical_features_on, tshark_filter, all_csv_file_path):
    # Create protocol folder
    if not os.path.exists(protocol_folder_path):
        os.makedirs(protocol_folder_path)

    # Read blacklisted features
    blacklisted_features = read_blacklisted_features(blacklist_file_path)

    # Read feature names from the filters folder
    csv_header = read_and_filter_feature_names(feature_names_file_path, blacklisted_features)

    # Write header row with feature names to csv files
    write_header_to_csv_files(csv_file_paths, csv_header)

    # List of classes (dict)
    list_of_classes = {}
    class_counter = 0

    # Loop through each pcap file in the provided folder
    for pcap_file_name in pcap_file_names:
        class_name = pcap_file_name.split('.pcap')[0]
        list_of_classes[class_counter] = class_name
        pcap_file_path = pcap_file_paths[class_counter]

        print("processing " + pcap_file_name + "...")

        # Prepare the tshark command
        tshark_cmd = ['tshark', '-n', '-r', pcap_file_path, '-T', 'fields']
        if tshark_filter:
            tshark_cmd.extend(['-Y', tshark_filter])
        for feature in csv_header[:-1]:
            tshark_cmd.extend(['-e', feature])
        tshark_cmd.extend(['-E', 'separator=/t'])

        # Process tshark output line by line
        with open(all_csv_file_path, 'w') as file:
            with subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, text=True) as proc:
                for line in proc.stdout:
                    fields = line.split('\t')
                    if len(fields) == len(csv_header) - 1:
                        if not all(entry == "" for entry in fields):  # Filter NaN values
                            fields = modify_dataset(fields)
                            fields.append(str(class_counter))
                            file.write(','.join(fields) + '\n')
        remove_duplicates_in_place(all_csv_file_path)
        split_file_into_thirds(all_csv_file_path, csv_file_paths)

        class_counter += 1

    os.remove(all_csv_file_path)
    
    print()

    # Write class list to file
    with open(classes_file_path, 'w') as json_file:
        json.dump(list_of_classes, json_file)

    # Determine features that are empty across all 3 batches
    print("determining empty fields...")
    remove_empty_fields_from_csv_files(csv_file_paths)

    if statistical_features_on:
        add_stat_features_to_csv_files(csv_file_paths)

    # Write remaining field names to file
    write_remaining_field_list_to_file(csv_file_paths, selected_field_list_file_path)

if __name__ == '__main__':
    if len(sys.argv) < 2: # check if at least one argument is provided
        print("Usage: python script.py arg1 arg2...")
        sys.exit(1)

    # Determine filters folder path
    filters_folder = os.path.join(os.path.dirname(__file__), "filters")
    if not os.path.exists(filters_folder):
        print("The 'filters' folder is missing")
        sys.exit(1)

    # Variables
    classifier_index = ""
    max_num_of_generations = 100
    num_of_iterations = 10
    num_of_packets_to_process = 0
    order_of_batches = []
    weights = [0.9,0.1]
    log_file_path = ""
    mode = ""
    folder = ""
    protocol = ""
    tshark_filter = ""
    run_number = 0
    batch_number = 0
    num_cores = multiprocessing.cpu_count() - 1 # Determine the number of CPU cores minus 1
    statistical_features_on = False

    # List of classifiers to test
    classifiers = [
        ("DT", DecisionTreeClassifier(random_state=42)),
        ("RF", RandomForestClassifier(random_state=42)),
        ("SVC", SVC(random_state=42)),
        ("LiSVC", LinearSVC(random_state=42, dual='auto', C=1.0, max_iter=10000)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42)),
        ("GNB", GaussianNB()),
        ("RIP", RIPPER()), # doesn't work with multi-class
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
        elif sys.argv[index] in ('-b', '--batch'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -b/--batch option")
                sys.exit(1)
            
            batch_number = int(sys.argv[index + 1])

            if batch_number == 1:
                order_of_batches = [1, 2, 3]
            elif batch_number == 2:
                order_of_batches = [1, 3, 2]
            elif batch_number == 3:
                order_of_batches = [2, 3, 1]
            else:
                print("Invalid batch number")
                sys.exit(1)

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
        elif sys.argv[index] in ('-n'):
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

            folder = fix_trailing_character(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-r', '--run-number'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -r/--run-number option")
                sys.exit(1)

            run_number = int(sys.argv[index + 1])
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-l', '--log'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -f/--folder option")
                sys.exit(1)

            log_file_path = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-s', '--statistics'):
            statistical_features_on = True
            index += 1
        elif sys.argv[index] in ('-m', '--mode'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -m/--mode option")
                sys.exit(1)

            mode = sys.argv[index + 1]
            index += 2  # Skip both the option and its value
        elif sys.argv[index] in ('-h', '--help'):
            print_usage()
            sys.exit(0)
        elif sys.argv[index] in ('-c', '--classifier'):
            if index + 1 >= len(sys.argv):
                print("Missing value for -c/--classifier option")
                sys.exit(1)

            classifier_index = sys.argv[index+1]
            index += 2  # Skip both the option and its value
        else:
            print(f"Unknown parameter! '{sys.argv[index]}'")
            sys.exit(1)

    # Set parameters and perform validation checks
    if log_file_path == "":
        log_file_path = (
            "packets_" + str(num_of_packets_to_process) +
            "_mode_" + str(mode) +
            "_clf_" + classifier_index +
            "_batch_" + str(batch_number) +
            ("_run_" + str(run_number) if run_number > 0 else "") +
            ".txt"
        )

    if folder == "":
        print("Workspace folder not given!")
        sys.exit(1)

    if not os.path.exists(folder):
        print(f"The folder {folder} is missing")
        sys.exit(1)

    pcap_folder_path = folder + "pcap"

    log_file_path = f'{folder}/{protocol}/{log_file_path}'
    if os.path.exists(log_file_path):
        os.remove(log_file_path) # Remove previous log file

    csv_file_paths = []
    classes_file_path = f'{folder}/{protocol}/classes.json'
    train_file_paths = []
    selected_field_list_file_path = f'{folder}/{protocol}/fields.txt'
    selected_field_list = []
    if os.path.exists(selected_field_list_file_path):
        with open(selected_field_list_file_path, 'r') as file:
            selected_field_list = file.readline().strip().split(',')

    # Run the mode
    if mode == 'extract':
        if not os.path.exists(pcap_folder_path):
            print("The 'pcap' folder is missing")
            sys.exit(1)

        blacklist_file_path = f'{filters_folder}/blacklist.txt'
        feature_names_file_path = f'{filters_folder}/{protocol}.txt'
        protocol_folder_path = f'{folder}{protocol}'
        for i in range(3):
            csv_file_paths.append(f'{folder}{protocol}/batch_{i+1}.csv')
        pcap_file_names = sorted([f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')])

        if len([f for f in os.listdir(pcap_folder_path) if f.endswith('.pcap')]) == 0:
            print("There are no pcap files in the 'pcap' folder")
            sys.exit(1)

        pcap_file_paths = [folder + "pcap/" + file_name for file_name in pcap_file_names]

        print("converting pcap files to csv format...\n")
        extract_features_from_pcap(
            blacklist_file_path, feature_names_file_path, protocol_folder_path,
            csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path,
            selected_field_list_file_path, statistical_features_on, tshark_filter,
            f'{folder}{protocol}/all.csv'
        )
    elif mode == 'report':
        report.run(folder + protocol, classifiers, classifier_index)
    elif mode == 'ga' or mode == 'aco' or mode == 'abc':
        if os.path.exists(log_file_path):
            print("There already exists a log file for the given configuration. Exiting...")
            sys.exit(0)

        train_file_paths.append(f'{folder}{protocol}/batch_{order_of_batches[0]}.csv')
        train_file_paths.append(f'{folder}{protocol}/batch_{order_of_batches[1]}.csv')
        test_file_path = f'{folder}{protocol}/batch_{order_of_batches[2]}.csv'
        fields_file_path = f'{folder}{protocol}/fields.txt'

        if mode == 'ga':
            log("running GA...\n", log_file_path)
            best_solution, best_fitness = ga.run(
                train_file_paths, int(classifier_index), classes_file_path,
                num_of_packets_to_process, num_of_iterations, weights,
                log_file_path, max_num_of_generations, fields_file_path,
                num_cores, classifiers
            )
        elif mode == 'aco':
            log("running ACO...\n", log_file_path)
            best_solution, best_fitness = aco.run(
                train_file_paths, int(classifier_index), classes_file_path,
                num_of_packets_to_process, num_of_iterations, weights,
                log_file_path, max_num_of_generations, fields_file_path,
                num_cores, classifiers
            )
        elif mode == 'abc':
            log("running ABC...\n", log_file_path)
            best_solution, best_fitness = bee.run(
                train_file_paths, int(classifier_index), classes_file_path,
                num_of_packets_to_process, num_of_iterations, weights,
                log_file_path, max_num_of_generations, fields_file_path,
                num_cores, classifiers
            )

        # Print best solution and the features selected
        sol_str = ''.join(map(str, best_solution))
        log(f"Best Solution:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}", log_file_path)
        log("\nSelected features:", log_file_path)
        for i in range(len(best_solution)):
            if best_solution[i] == 1:
                log(selected_field_list[i], log_file_path)

        # Print the classification result on test data using selected features
        log("", log_file_path)
        log("Selected feature-set results:", log_file_path)
        ml.classify_after_filtering(best_solution, train_file_paths, test_file_path, int(classifier_index), log_file_path, classifiers, True)
        
        # Print the classification result on test data using all features
        log("All feature-set results:", log_file_path)
        ml.classify_after_filtering(best_solution, train_file_paths, test_file_path, int(classifier_index), log_file_path, classifiers, False)
    else:
        print("Unknown entry for the mode")
