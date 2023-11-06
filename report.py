from collections import Counter
import os
import re
import numpy as np
import sys
import matplotlib.pyplot as plt
from itertools import takewhile

def process_accuracies(batches_data, clfs, feature_key):
    err_l, main_acc, err_u = [], [], []
    for clf in clfs:
        acc = []
        for i in range(3):
            acc.append([batch_data[feature_key] for batch_data in batches_data if batch_data['classifier'] == clf and batch_data['batch_number'] == (i+1)][0])
        acc = sorted(acc)        
        err_l.append((acc[1] - acc[0])*100)
        main_acc.append(acc[1]*100)
        err_u.append((acc[2] - acc[1])*100)
    return err_l, main_acc, err_u

def plot(batches_data, clfs, folder, classifiers, mode):
    ga_data = process_accuracies(batches_data, clfs, 'selected_features_f1')
    nonga_data = process_accuracies(batches_data, clfs, 'all_features_f1')
    labels = [classifiers[int(clf)][0] for clf in clfs]

    fig, ax = plt.subplots(figsize=(3,5))
    ind, width = np.arange(len(clfs)), 0.30

    for offset, color, data in zip([0.15, 0.15 + width], ['#1C6CAB', '#FF7311'], [ga_data, nonga_data]):
        ax.errorbar(ind + offset, data[1], yerr=[data[0], data[2]], mec=color, ecolor=color, fmt='o', capsize=4, mew=5)

    ax.set_xticks(ind + 0.10 + width / 2)
    ax.set_xticklabels(labels, fontsize=15)
    plt.subplots_adjust(left=0.17, right=0.99, top=0.92, bottom=0.06) # Modify top value
    plt.ylim([0, 100])
    plt.yticks(fontsize=15)

    # Set the title for the figure
    protocol_name = batches_data[0]['file_path'].split('/')[1].upper().replace('_', ' & ')

    # Set the title for the figure
    ax.set_title(protocol_name, fontsize=16)

    plt.savefig(f"{folder}/plot_" + mode + ".pdf", format='pdf', dpi=1000, bbox_inches='tight') # Add bbox_inches

def report(batches_data, clfs, folder, mode):
    file_path = f"{folder}/report_" + mode + ".txt"
    if os.path.exists(file_path):
        os.remove(file_path)

    selected_features_all_clfs = []

    for clf in clfs:
        # Determine the maximum accuracy
        selected_features_f1_max = 0
        all_features_f1_max = 0
        files_f1_max = []
        number_of_batches = len(set(entry['batch_number'] for entry in batches_data))
        for batch_number in range(number_of_batches): # for each batch
            validation_f1_max_index = max((i for i, batch_data in enumerate(batches_data) if batch_data['classifier'] == clf and batch_data['batch_number'] == (batch_number+1)), key=lambda index: batches_data[index]['validation_f1'])
            selected_features_f1_max += batches_data[validation_f1_max_index]['selected_features_f1']
            all_features_f1_max += batches_data[validation_f1_max_index]['all_features_f1']
            files_f1_max.append(os.path.basename(batches_data[validation_f1_max_index]['file_path']))
        selected_features_f1_max /= 3
        all_features_f1_max /= 3

        # Determine the features that were selected the most
        selected_features = [feature for batch_data in batches_data for feature in batch_data['selected_features'] if batch_data['classifier'] == clf]
        selected_features_all_clfs.extend(selected_features)
        sorted_items = sorted(Counter(selected_features).items(), key=lambda x: x[1], reverse=True)

        # Write to file
        with open(file_path, 'a') as file:
            # Wrtie the classifier name
            file.write(f"Classifier: {clf}\n")
            
            # Write the average accuracy
            file.write(f"\tMaximum F1 of selected features: {selected_features_f1_max}\n")
            file.write(f"\tMaximum F1 of all features:\t {all_features_f1_max}\n")
            file.write(f"\tMaximum F1 yielding files:\n")
            for file_f1_max in files_f1_max:
                file.write(f"\t\t{file_f1_max}\n")
            file.write("\n")

            # Write the features that were selected the most
            file.write(f"\tFeatures selected the most:\n")
            for item, count in sorted_items:
                file.write(f"\t\t{count}\t{item}\n")
            file.write(f"\n\n\n")
    
    sorted_items_all_clfs = sorted(Counter(selected_features_all_clfs).items(), key=lambda x: x[1], reverse=True)
    with open(file_path, 'a') as file:
            # Write the features that were selected the most
            file.write(f"Across all Classifiers:\n")
            file.write(f"\tFeatures selected the most:\n")
            for item, count in sorted_items_all_clfs:
                file.write(f"\t\t{count}\t{item}\n")

def run(folder, classifiers, classifier_indices):
    file_names = sorted(os.listdir(folder), key=lambda s: [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)])
    full_paths = [os.path.join(folder, file) for file in file_names if file.startswith("packets_") and file.endswith(".txt")]

    modes = list(set([os.path.basename(path).split('_')[3] for path in full_paths]))
    for mode in modes:
        batches_data = []
        for path in full_paths:
            filename = os.path.basename(path)
            if filename.split('_')[3] == mode:
                with open(path, 'r') as f:
                    content = f.readlines()
                
                # Process file data
                validation_f1 = float(next(line.split()[-1] for line in content if "Best Solution:" in line))
                selected_features = [line.strip() for line in takewhile(lambda x: x.strip(), content[next(i for i, line in enumerate(content) if "Selected features:" in line) + 1:])]
                f1_values = [float(line.split()[-1]) for line in content if "F1" in line]
                batches_data.append({
                    "mode": filename.split('_')[3],
                    "classifier": int(filename.split('_')[5]),
                    "batch_number": int(filename.split('_')[7]),
                    "run_number": int(filename.split('_')[9].split(".")[0]),
                    "selected_features": selected_features,
                    "validation_f1": validation_f1,
                    "selected_features_f1": f1_values[-2],
                    "all_features_f1": f1_values[-1],
                    "file_path": path
                })

        if classifier_indices == "":
            clfs = sorted(list(set(batch_data['classifier'] for batch_data in batches_data)))
        else:
            clfs = list(map(int, classifier_indices.split(',')))
            for clf in clfs:
                if clf not in list(set(batch_data['classifier'] for batch_data in batches_data)):
                    print(f"Classifier {clf} not found in the data. Exiting...")
                    sys.exit(1)
        print("plotting the diagram...")
        plot(batches_data, clfs, folder, classifiers, mode)
        print("generating the report...")
        report(batches_data, clfs, folder, mode)