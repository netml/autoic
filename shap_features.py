import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
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
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Get SHAP values array
    shap_values = shap_values.values if hasattr(shap_values, 'values') else np.array(shap_values)

    # Compute average SHAP values per feature
    average_shap_values = np.mean(shap_values, axis=0)
    feature_names = X.columns.tolist()

    # Log full SHAP values
    all_scr = {feature: shap_value for feature, shap_value in zip(feature_names, average_shap_values)}
    logging.info("SHAP feature values:")
    logging.info(all_scr)

    # Select features with positive average SHAP value
    positive_features = [
        feature for feature, shap_value in zip(feature_names, average_shap_values) if (shap_value > 0).any()
    ]

    # Log positive features
    logging.info("Features with Positive Average SHAP Values:")
    logging.info(f"Total positive features selected: {len(positive_features)} out of {len(feature_names)}")
    logging.info(positive_features)

    return positive_features

def generate_new(features, X_train, y_train, X_test, y_test):
    X_train_new = X_train[features]
    X_test_new = X_test[features]

    clf = DecisionTreeClassifier()
    clf.fit(X_train_new, y_train)

    y_pred = clf.predict(X_test_new)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test accuracy after SHAP feature selection: {acc * 100:.2f}%")
    logging.info("Classification Report (SHAP features):")
    logging.info(classification_report(y_test, y_pred, zero_division=0))

    return features  # return selected features

def run(train_file_path, test_file_path, log_folder):
    logfile = os.path.join(log_folder, 'shap.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')

    # Load datasets
    X_train, y_train = load_file(train_file_path)
    X_test, y_test = load_file(test_file_path)

    # Train full-feature model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test accuracy before SHAP: {acc * 100:.2f}%")
    logging.info("Classification Report (All features):")
    logging.info(classification_report(y_test, y_pred, zero_division=0))

    # SHAP-based feature selection
    features = shap_feat(clf, X_train)
    selected_features = generate_new(features, X_train, y_train, X_test, y_test)

    return selected_features