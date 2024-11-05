import sys
import os
import textwrap


def log(message, log_file_path):
    print(message)
    with open(log_file_path, 'a') as file:
        file.write(message + "\n")

def fix_trailing_character(input_string):
    return input_string.rstrip('/') + '/'  # Remove trailing '/' and add it back

def check_path_exists(path, path_name, is_folder=True):
    if path == "":
        print(f"{path_name} not provided!")
        sys.exit(1)
    if not os.path.exists(path):
        folder_or_file = 'folder' if is_folder else 'file'
        print(f"The {path_name} {folder_or_file} ({path}) is missing.")
        sys.exit(1)

def generate_specific_combinations(n):
    combinations = []
    for i in range(1, n + 1):
        combination = [j for j in range(1, n + 1) if j != i]
        combination.sort()
        combination.append(i)
        combinations.append(combination)
    return combinations

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
                                    6: KNN
                                    7: Logistic Regression

    -b, --num-of-batches <number>   Specify the k-fold cross-validation number. The value should
                                    be an integer. This option is set to 3 by default.

    -h, --help                      Display this help message and exit.

    Note: Replace <placeholders> with actual values without the angle brackets.""")
    print(message)
