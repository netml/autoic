from collections import defaultdict
import hashlib
import tempfile
import pandas as pd
import subprocess
import csv
import sys
import re
import json
import statistics
import os
import libraries
import shap_features
import shutil

def read_blacklisted_features(blacklist_check, blacklist_file_path):
    if blacklist_check:
        try:
            with open(blacklist_file_path, 'r') as f:
                return f.read().splitlines()
        except FileNotFoundError:
            return []  # return empty list if file not found
    else:
        return []  # return empty list if blacklist_check is false

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
        stats.extend([min(column_values), max(column_values), statistics.mean(column_values),
                      statistics.stdev(column_values), statistics.mode(column_values)])

    csv_data = [new_header]

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

def remove_empty_columns_from_csv_file(all_csv_file_path, chunk_size=10000):
    non_unique_columns = set()
    first_chunk = True
    label_column = None
    no_of_columns = 0

    for chunk in pd.read_csv(all_csv_file_path, chunksize=chunk_size, low_memory=False):
        if first_chunk:
            # Initialize non_unique_columns for all columns except the label column
            no_of_columns = chunk.columns.size
            label_column = chunk.columns[-1]
            for col in chunk.columns[:-1]:
                if chunk[col].nunique() > 1:
                    non_unique_columns.add(col)
            first_chunk = False
        else:
            for col in chunk.columns[:-1]:
                if col not in non_unique_columns and chunk[col].nunique() > 1:
                    non_unique_columns.add(col)

    # Add the label column to the non_unique_columns set
    ordered_columns = sorted(list(non_unique_columns))
    ordered_columns.append(label_column)

    if len(ordered_columns) == 1:
        print("all fields except the label column are redundant, skipping...")
        return

    print(f"keeping {len(ordered_columns) - 1}/{no_of_columns - 1} fields...")

    # Create a temporary file to store the filtered data
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file_path = temp_file.name

    # Flag to check if it's the first chunk
    is_first_chunk = True

    # Process the CSV in chunks and write to the temporary file
    for chunk in pd.read_csv(all_csv_file_path, chunksize=chunk_size, low_memory=False):
        filtered_chunk = chunk[ordered_columns]
        filtered_chunk.to_csv(temp_file_path, mode='a', index=False, header=is_first_chunk)
        if is_first_chunk:
            is_first_chunk = False

    temp_file.close()

    # Replace the original file with the temporary file
    os.replace(temp_file_path, all_csv_file_path)

def write_header_to_csv_file(csv_file_path, feature_names):
    with open(csv_file_path, 'w') as f:
        csv_line = ','.join(feature_names)
        f.write(f"{csv_line}\n")

def write_extracted_field_list_to_file(csv_file_path, extracted_field_list_file_path):
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        extracted_field_list = next(csv_reader)

    with open(extracted_field_list_file_path, 'w') as file:
        for field in extracted_field_list[:-1]:
            file.write(f"{field}\n")

def consistent_numerical_hash(token):
    # Convert the token to bytes if it's a string
    if isinstance(token, str):
        token = token.encode('utf-8')

    # Use SHA-256 hashing algorithm
    hash_object = hashlib.sha256()
    hash_object.update(token)

    # Convert the hash to a large integer
    return int.from_bytes(hash_object.digest(), 'big')

def convert_token(token):
    token = token.strip()
    if token:
        if re.match(r'^0x[0-9a-fA-F]+$', token):  # Hexadecimal
            return int(token, 16)
        elif is_numeric(token):  # Decimal
            return float(token)
        else:  # String
            return consistent_numerical_hash(token)
    else:
        return -1.0

def modify_dataset(fields):
    for i, cell in enumerate(fields):
        if cell:
            tokens = cell.split(',')
            total = sum(convert_token(token) for token in tokens)
            fields[i] = str(total % 0xFFFFFFFF)
        else:
            fields[i] = '-1.0'
    return fields

def is_numeric(token):
    try:
        float(token)
        return True
    except ValueError:
        return False

def remove_duplicate_rows_from_csv_file(file_path):
    seen = set()
    temp_file_path = file_path + ".tmp"

    try:
        # Read unique lines and write to a temporary file
        with open(file_path, 'r') as file, open(temp_file_path, 'w') as temp_file:
            for line in file:
                stripped_line = line.strip()  # Optionally strip whitespace
                if stripped_line not in seen:
                    temp_file.write(line)
                    seen.add(stripped_line)

        # Rename the temporary file to original file path
        os.replace(temp_file_path, file_path)
    except IOError as e:
        print(f"An error occurred: {e}")
        # Optionally remove the temporary file if an error occurred
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def count_lines_in_file(file_path):
    with open(file_path, 'r') as file:
        line_count = sum(1 for _ in file)
    return line_count

def split_csv(all_csv_file_path, output_files, chunksize=10000):
    # Check if the number of output files matches the required split count
    split_count = len(output_files)
    if split_count < 2:
        raise ValueError("At least two output files must be specified.")

    # First pass: Count the occurrences of each label
    label_counts = defaultdict(int)
    with pd.read_csv(all_csv_file_path, chunksize=chunksize, iterator=True) as reader:
        for chunk in reader:
            labels = chunk.iloc[:, -1].value_counts()  # Assuming label is in the last column
            for label, count in labels.items():
                label_counts[label] += count

    # Remove labels where the count is less than the number of splits
    label_counts = {label: count for label, count in label_counts.items() if count >= split_count}

    # Initialize file handles and write headers
    file_handles = {file: open(file, 'w') for file in output_files}
    with open(all_csv_file_path, 'r') as reader:
        header = reader.readline()
        for f in file_handles.values():
            f.write(header)

    # Calculate the number of lines to write to each file
    file_index = {label: [] for label in label_counts}
    for label in label_counts:
        max_lines_per_group = label_counts[label] // split_count
        file_index[label] = [max_lines_per_group for _ in range(split_count)]
        for i in range(label_counts[label] % split_count):
            file_index[label][i] += 1

    # Second pass: Distribute rows to each file
    with open(all_csv_file_path, 'r') as reader:
        reader.readline()  # Skip the header row
        for line in reader:
            label = int(line.strip().split(',')[-1])
            if label in label_counts:
                for i, index in enumerate(file_index[label]):
                    if index > 0:
                        file_handles[output_files[i]].write(line)
                        file_index[label][i] -= 1
                        break

    # Close all file handles
    for f in file_handles.values():
        f.close()

def replace_csv_file(all_csv_file_path, shap_csv_file_path):
    # Remove the file at all_csv_file_path if it exists
    if os.path.exists(all_csv_file_path):
        os.remove(all_csv_file_path)
    
    shutil.move(shap_csv_file_path, all_csv_file_path)

def create_batch_files(num_of_batches, split_file_paths, batch_file_paths, libraries):
    if not all(os.path.exists(file_path) for file_path in [file_path for sublist in batch_file_paths for file_path in sublist]):
        print("Creating the batch files...")
        # Create test files
        for i in range(num_of_batches):
            # Create test files
            libraries.copy_file(split_file_paths[i], batch_file_paths[1][i])

            # Create train files
            train_files = [file_path for j, file_path in enumerate(split_file_paths) if j != i]
            header_written = False

            with open(batch_file_paths[0][i], 'w') as train_file:
                for file_path in train_files:
                    with open(file_path, 'r') as file:
                        for line_num, line in enumerate(file):
                            # Write header only once
                            if line_num == 0:
                                if not header_written:
                                    train_file.write(line)
                                    header_written = True
                                continue
                            # Write the remaining lines
                            train_file.write(line)

def original(blacklist_check, blacklist_file_path, feature_names_file_path, protocol_folder_path, split_file_paths,
        pcap_file_names, pcap_file_paths, classes_file_path, extracted_field_list_file_path,
        statistical_features_on, tshark_filter, all_csv_file_path, batch_file_paths, num_of_batches):

    # Create protocol folder if it doesn't exist
    if not os.path.exists(protocol_folder_path):
        os.makedirs(protocol_folder_path)

    # Check if the main CSV file exists; if not, process PCAP files
    if not os.path.exists(all_csv_file_path):
        print("converting pcap files to csv format...")

        # Read blacklisted features
        blacklisted_features = read_blacklisted_features(blacklist_check, blacklist_file_path)

        # Read feature names from the filters folder
        csv_header = read_and_filter_feature_names(feature_names_file_path, blacklisted_features)

        # Write header rows to CSV file
        write_header_to_csv_file(all_csv_file_path, csv_header)

        # List of classes (dict)
        list_of_classes = {}

        # Loop through each PCAP file
        max_length = max(len(pcap_file_name) for pcap_file_name in pcap_file_names) + (len(pcap_file_names) * 2) + 7
        for class_counter, pcap_file_name in enumerate(pcap_file_names):
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
                            if not all(entry == "" for entry in fields):  # Filter NaN values
                                fields = modify_dataset(fields)
                                fields.append(str(class_counter))
                                file.write(','.join(fields) + '\n')

        print("removing redundant fields...")
        remove_empty_columns_from_csv_file(all_csv_file_path)

        print("removing duplicate rows...")
        remove_duplicate_rows_from_csv_file(all_csv_file_path)

        if statistical_features_on:
            print("adding statistical features...")
            add_stat_features_to_csv_file(all_csv_file_path)

    # Check if split files exist; if not, create them
    if not all(os.path.exists(file_path) for file_path in split_file_paths):
        print("creating the split files...")
        split_csv(all_csv_file_path, split_file_paths)

    # Create batch files
    create_batch_files(num_of_batches, split_file_paths, batch_file_paths, libraries)

    # Write extracted field list to files if they don't exist
    if not os.path.exists(extracted_field_list_file_path):
        print("generating feature sets...")
        write_extracted_field_list_to_file(all_csv_file_path, extracted_field_list_file_path)

    # Write classes JSON file if it doesn't exist
    if not os.path.exists(classes_file_path):
        print("generating the classes list...")
        list_of_classes = {i: pcap_file_name.split('.pcap')[0] for i, pcap_file_name in enumerate(pcap_file_names)}
        json_data = json.dumps(list_of_classes, ensure_ascii=False)  # Convert list_of_classes to JSON string
        with open(classes_file_path, 'w', encoding='utf-8') as json_file:
            json_file.write(json_data)

def shap(protocol_folder_path, split_file_paths, extracted_field_list_file_path, all_csv_file_path, shap_csv_file_path,
         shap_fold_size, batch_file_paths, num_of_batches):

    # Check if the protocol folder exists
    if not os.path.exists(protocol_folder_path):
        print("The protocol folder does not exist.")
        sys.exit(1)

    # Check if the main CSV file exists
    if not os.path.exists(all_csv_file_path):
        print("The main CSV file does not exist.")
        sys.exit(1)

    # Run SHAP feature extraction if enabled
    if not os.path.exists(shap_csv_file_path):
        print("running SHAP feature extraction...")
        shap_features.run(all_csv_file_path, shap_csv_file_path, protocol_folder_path, shap_fold_size)

    # Check if split files exist; if not, create them
    if not all(os.path.exists(file_path) for file_path in split_file_paths):
        print("generating SHAP batch files...")
        split_csv(shap_csv_file_path, split_file_paths)

    # Create batch files
    create_batch_files(num_of_batches, split_file_paths, batch_file_paths, libraries)

    # Write extracted field list to files if they don't exist
    if not os.path.exists(extracted_field_list_file_path):
        write_extracted_field_list_to_file(shap_csv_file_path, extracted_field_list_file_path)
