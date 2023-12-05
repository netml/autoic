import pandas as pd
import subprocess
import csv
import sys
import os
import re
import json
import statistics

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

def convert_token(token):
    token = token.strip()

    if re.match(r'^0x[0-9a-fA-F]+$', token):
        return int(token, 16)
    elif not is_numeric(token):
        return hash(token)
    else:
        return float(token)

def modify_dataset(fields):
    for j, cell in enumerate(fields):
        if cell:
            tokens = cell.split(',')
            total = sum(convert_token(token) for token in tokens)
            fields[j] = str(total % 0xFFFFFFFF)
        else:
            fields[j] = '-1'
    return fields

def is_numeric(token):
    try:
        float(token)
        return True
    except ValueError:
        return False

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

def run(blacklist_file_path, feature_names_file_path, protocol_folder_path, csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, selected_field_list_file_path, statistical_features_on, tshark_filter, all_csv_file_path):
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
