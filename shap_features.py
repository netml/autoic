import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
import shap
import os
import logging

def load_file(fpath):
    data = pd.read_csv(fpath)
    target_column = 'label'
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def shap_feat(model, X):
    # Create an explainer for the model
    explainer = shap.Explainer(model)

    # Get SHAP values
    shap_values = explainer(X)

    # Ensure shap_values is a NumPy array and access the values
    shap_values = shap_values.values if hasattr(shap_values, 'values') else np.array(shap_values)

    # Compute the average SHAP values across instances
    average_shap_values = np.mean(shap_values, axis=0)

    # Get feature names
    feature_names = X.columns.tolist()

    # Create a dictionary for SHAP values of each feature
    all_scr = {feature: shap_value for feature, shap_value in zip(feature_names, average_shap_values)}

    # Log the SHAP feature values
    logging.info("SHAP feature values:")
    logging.info(all_scr)

    # Get the positive features based on average SHAP values
    positive_features = [
        feature for feature, shap_value in zip(feature_names, average_shap_values) if (shap_value > 0).any()
    ]

    # Log positive features
    logging.info("Features with Positive Average SHAP Values:")
    logging.info(f"Total positive features selected: {len(positive_features)} out of total features: {len(feature_names)},"
                 f"{len(positive_features)}/{len(feature_names)}")
    logging.info(positive_features)

    # Return the positive features
    return positive_features

def generate_new(features, X, y, folds, shap_csv_file_path):
    X_new = X[features].assign(label=y)  # Assigning y as a new column 'Label'
    X_new.to_csv(shap_csv_file_path, index=False)
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X_new.drop(columns=['label']), y, cv=folds)  # Dropping the label column for training
    for i, score in enumerate(scores):
        logging.info(f"Fold {i+1} accuracy: {score}")
    logging.info(f"Mean accuracy: {scores.mean() * 100:.2f}%")
    # print(f"Mean accuracy after shap: {scores.mean() * 100:.2f}%")
    return X_new, clf, y

def clf_rep(clf, X_new, y, folds):
    predicted_labels = cross_val_predict(clf, X_new, y, cv=folds)
    report = classification_report(y, predicted_labels)
    logging.info("Classification Report:")
    logging.info(report)
    # print("Classification Report:")
    # print(report)

def classify(X, y, folds):
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, y, cv=folds)
    for i, score in enumerate(scores):
        logging.info(f"Fold {i+1} accuracy: {score}")
    logging.info(f"Mean accuracy: {scores.mean() * 100:.2f}%")
    # print(f"Mean accuracy before shap: {scores.mean() * 100:.2f}%")
    model = clf.fit(X, y)
    return model, clf

def run(all_csv_file_path, shap_csv_file_path, folder_path, folds):
    logfile = os.path.join(folder_path, 'shap' + '.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')
    X, y = load_file(all_csv_file_path)
    clf, model = classify(X, y, folds)
    clf_rep(clf, X, y, folds)
    features = shap_feat(model, X)
    X_new, clf2, y = generate_new(features, X, y, folds, shap_csv_file_path)
    clf_rep(clf2, X_new, y, folds)
