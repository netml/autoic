from collections import defaultdict
import pandas as pd
import subprocess
import csv
import sys
import os
import re
import json
import statistics
import os
import numpy as np

def read_blacklisted_features(blacklist_check, blacklist_file_path):
    if (blacklist_check):
        try:
            with open(blacklist_file_path, 'r') as f:
                return f.read().splitlines()
        except FileNotFoundError:
            return [] # return empty list if file not found
    else:
        return [] # return empty list if blacklist_check is false

def read_and_filter_feature_names(feature_names_file_path, blacklisted_features):
    try:
        with open(feature_names_file_path, 'r') as f:
            feature_names = [feature for feature in f.read().splitlines() if feature not in blacklisted_features]
            feature_names.append('label')
        return feature_names
    except FileNotFoundError:
        print(f"The file '{feature_names_file_path}' was not found.")
        sys.exit(1)

def add_stat_features_to_csv_file(file_path):
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

def remove_empty_fields_from_csv_file(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, low_memory=False)

    # Find column names
    common_columns = set(df.columns[:-1])
    columns_to_remove = set()

    for col in common_columns:
        unique_values = set(df[col].unique())
        if len(unique_values) == 1: # If there is only one unique value, the column is redundant
            columns_to_remove.add(col)

    if len(columns_to_remove) == 0: # check if there are no empty fields
        print(f"there do not exist redundant fields, keeping all {len(common_columns)} fields...")
    elif len(columns_to_remove) < len(common_columns): # check if we are not removing all columns
        print(f"removing {len(columns_to_remove)}/{len(common_columns)} redundant fields...")
        df = df.drop(columns=columns_to_remove, errors='ignore')
        df.to_csv(csv_file_path, index=False)
    else:
        print(f"all {len(columns_to_remove)} fields are redundant, skipping...")

def write_header_to_csv_file(csv_file_path, feature_names):
    with open(csv_file_path, 'w') as f:
        csv_line = ','.join(feature_names)
        f.write(f"{csv_line}\n")

def write_remaining_field_list_to_file(csv_file_path, selected_field_list_file_path):
    selected_field_list = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        selected_field_list = next(csv_reader)

    with open(selected_field_list_file_path, 'w') as file:
        file.write(','.join(selected_field_list[:-1]))

def convert_token(token):
    token = token.strip()
    if token:
        if re.match(r'^0x[0-9a-fA-F]+$', token):
            return int(token, 16)
        elif not is_numeric(token):
            return hash(token)
        else:
            return float(token)
    else:
        return -1

def modify_dataset(fields):
    for j, cell in enumerate(fields):
        if cell:
            tokens = cell.split(',')
            total = sum(convert_token(token) for token in tokens)
            fields[j] = str(total % 0xFFFFFFFF)
        else:
            fields[j] = '-1.0'
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

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for _ in file)
    return line_count

def split_csv_by_label(input_csv, output_files, chunksize=10000):
    # Check if the number of output files matches the required split count
    split_count = len(output_files)
    if split_count < 1:
        raise ValueError("At least one output file must be specified.")

    # First pass: Count the occurrences of each label
    label_counts = defaultdict(int)
    with pd.read_csv(input_csv, chunksize=chunksize) as reader:
        for chunk in reader:
            labels = chunk['label'].value_counts()
            for label, count in labels.items():
                label_counts[label] += count

    # Calculate the number of rows of each label per file
    label_rows_per_file = {label: np.array_split(range(count), split_count) for label, count in label_counts.items()}

    # Track the next row index to read for each label in each file
    next_row_index = {label: [0]*split_count for label in label_counts}

    # Initialize file pointers for each output file
    files = {file: open(file, 'w', newline='') for file in output_files}

    # Write headers
    headers = pd.read_csv(input_csv, nrows=0).to_csv(index=False)
    for file in files.values():
        file.write(headers)

    # Second pass: Distribute rows to files
    with pd.read_csv(input_csv, chunksize=chunksize) as reader:
        for chunk in reader:
            for label, rows in label_rows_per_file.items():
                label_chunk = chunk[chunk['label'] == label]
                for i in range(split_count):
                    # Get the indices for the current chunk
                    start_idx = next_row_index[label][i]
                    end_idx = start_idx + len(rows[i])
                    next_row_index[label][i] = end_idx

                    # Select rows for the current chunk
                    if start_idx < len(label_chunk):
                        selected_rows = label_chunk.iloc[start_idx:end_idx]
                        selected_rows.to_csv(files[output_files[i]], index=False, header=False, mode='a')

    # Close the file pointers
    for file in files.values():
        file.close()

def run(blacklist_check, blacklist_file_path, feature_names_file_path, protocol_folder_path, csv_file_paths, pcap_file_names, pcap_file_paths, classes_file_path, selected_field_list_file_path, statistical_features_on, tshark_filter, all_csv_file_path):
    # Create protocol folder
    if not os.path.exists(protocol_folder_path):
        os.makedirs(protocol_folder_path)

    # Read blacklisted features
    blacklisted_features = read_blacklisted_features(blacklist_check, blacklist_file_path)

    # Read feature names from the filters folder
    csv_header = read_and_filter_feature_names(feature_names_file_path, blacklisted_features)

    # Write header row with feature names to csv files
    write_header_to_csv_file(all_csv_file_path, csv_header)

    # List of classes (dict)
    list_of_classes = {}
    class_counter = 0

    # Loop through each pcap file in the provided folder
    max_length = max(len(pcap_file_name) for pcap_file_name in pcap_file_names) + (len(pcap_file_names) * 2) + 7
    for pcap_file_name in pcap_file_names:
        class_name = pcap_file_name.split('.pcap')[0]
        list_of_classes[class_counter] = class_name
        pcap_file_path = pcap_file_paths[class_counter]

        print(f"{'[' + str(class_counter+1) + '/' + str(len(pcap_file_names)) + '] ' + pcap_file_name + '...':<{max_length}}\r", end='')

        # Prepare the tshark command
        tshark_cmd = ['tshark', '-n', '-r', pcap_file_path, '-T', 'fields']
        if tshark_filter:
            tshark_cmd.extend(['-Y', tshark_filter])
        for feature in csv_header[:-1]:
            tshark_cmd.extend(['-e', feature])
        tshark_cmd.extend(['-E', 'separator=/t'])

        # Process tshark output line by line
        with open(all_csv_file_path, 'a') as file:
            with subprocess.Popen(tshark_cmd, stdout=subprocess.PIPE, text=True) as proc:
                for line in proc.stdout:
                    fields = line.split('\t')
                    if len(fields) == len(csv_header) - 1:
                        if not all(entry == "" for entry in fields): # Filter NaN values
                            fields = modify_dataset(fields)
                            fields.append(str(class_counter))
                            file.write(','.join(fields) + '\n')
        class_counter += 1

    print("checking for redundant fields...")
    remove_empty_fields_from_csv_file(all_csv_file_path)

    print("removing duplicate rows...")
    remove_duplicates_in_place(all_csv_file_path)

    # Add statistical features to csv files
    if statistical_features_on:
        print("adding statistical features...")
        add_stat_features_to_csv_file(all_csv_file_path)

    split_csv_by_label(all_csv_file_path, csv_file_paths)

    # Write class list to file
    with open(classes_file_path, 'w') as json_file:
        json.dump(list_of_classes, json_file)

    # Write remaining field names to file
    write_remaining_field_list_to_file(all_csv_file_path, selected_field_list_file_path)

    os.remove(all_csv_file_path)
