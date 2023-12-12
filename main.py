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
import multiprocessing
import sys
import os
import ga
import ml, aco, bee
import report
import extract
import textwrap

def print_usage():
    message = textwrap.dedent("""\
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

    -nb, --no-blacklist             Disable blacklist check. This option does not require a value.
                                    If specified, the tool will not check the blacklist file for
                                    eliminating features. Blacklist check is enabled by default.

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
                                    
    -h, --help                      Display this help message and exit.

    Note: Replace <placeholders> with actual values without the angle brackets.""")
    print(message)

def fix_trailing_character(input_string):
    return input_string.rstrip('/') + '/'  # Remove trailing '/' and add it back

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
    order_of_batches = [[1, 2, 3], [1, 3, 2], [2, 3, 1]]
    weights = [0.9,0.1]
    mode = ""
    folder = ""
    protocol = ""
    tshark_filter = ""
    run_number = 0
    batch_number = 0
    num_cores = multiprocessing.cpu_count() - 1 # Determine the number of CPU cores minus 1
    statistical_features_on = False
    blacklist_check = True

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

            order_of_batches = []
            if batch_number == 1:
                order_of_batches.append([1, 2, 3])
            elif batch_number == 2:
                order_of_batches.append([1, 3, 2])
            elif batch_number == 3:
                order_of_batches.append([2, 3, 1])
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
        elif sys.argv[index] in ('-nb', '--no-blacklist'):
            blacklist_check = False
            index += 1
        else:
            print(f"Unknown parameter! '{sys.argv[index]}'")
            sys.exit(1)

    if folder == "":
        print("Workspace folder not given!")
        sys.exit(1)

    if not os.path.exists(folder):
        print(f"The folder {folder} is missing")
        sys.exit(1)

    pcap_folder_path = folder + "pcap"

    csv_file_paths = []
    classes_file_path = f'{folder}/{protocol}/classes.json'
    extracted_field_list_file_path = f'{folder}/{protocol}/fields.txt'

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

        print("converting pcap files to csv format...")

        extract.run(
            blacklist_check, blacklist_file_path, feature_names_file_path, protocol_folder_path,
            csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path,
            extracted_field_list_file_path, statistical_features_on, tshark_filter,
            f'{folder}{protocol}/all.csv'
        )

        print("done...")
    elif mode == 'report':
        if classifier_index == "":
            print("Classifier index not given!")
            sys.exit(1)

        report.run(folder + protocol, classifiers, classifier_index)
    elif mode == 'ga' or mode == 'aco' or mode == 'abc':
        if classifier_index == "":
            print("Classifier index not given!")
            sys.exit(1)

        if batch_number == 0:
            batch_number = 1
        for order_of_batch in order_of_batches:
            log_file_path = (
                "packets_" + str(num_of_packets_to_process) +
                "_mode_" + str(mode) +
                "_clf_" + classifier_index +
                "_batch_" + str(batch_number) +
                ("_run_" + str(run_number) if run_number > 0 else "") +
                ".txt"
            )

            log_file_path = f'{folder}/{protocol}/{log_file_path}'
            if os.path.exists(log_file_path):
                os.remove(log_file_path) # Remove previous log file

            train_file_paths = []
            train_file_paths.append(f'{folder}{protocol}/batch_{order_of_batch[0]}.csv')
            train_file_paths.append(f'{folder}{protocol}/batch_{order_of_batch[1]}.csv')
            test_file_path = f'{folder}{protocol}/batch_{order_of_batch[2]}.csv'
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

            # Read the extracted field list
            extracted_field_list = []
            if os.path.exists(extracted_field_list_file_path):
                with open(extracted_field_list_file_path, 'r') as file:
                    extracted_field_list = [line.strip() for line in file.readlines()]

            # Print best solution and the features selected
            sol_str = ''.join(map(str, best_solution))
            log(f"Best Solution:\t[{sol_str}]\t[{sol_str.count('1')}/{len(sol_str)}]\tFitness: {best_fitness}", log_file_path)
            log("\nSelected features:", log_file_path)
            for i in range(len(best_solution)):
                if best_solution[i] == 1:
                    log(extracted_field_list[i], log_file_path)

            # Print the classification result on test data using selected features
            log("", log_file_path)
            log("Selected feature-set results:", log_file_path)
            ml.classify_after_filtering(best_solution, train_file_paths, test_file_path, int(classifier_index), log_file_path, classifiers, True)
            
            # Print the classification result on test data using all features
            log("All feature-set results:", log_file_path)
            ml.classify_after_filtering(best_solution, train_file_paths, test_file_path, int(classifier_index), log_file_path, classifiers, False)

            batch_number += 1
    else:
        print("Unknown entry for the mode")
