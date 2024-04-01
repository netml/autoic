import sys
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
    explainer = shap.Explainer(model)
    instances = X
    shap_values = explainer.shap_values(instances)
    average_shap_values = np.mean(shap_values, axis=0)
    feature_names = X.columns.tolist()
    all_scr = {feature: shap_value for feature, shap_value in zip(feature_names, average_shap_values)}
    logging.info("SHAP feature values:")
    logging.info(all_scr)
    positive_features = [feature for feature, shap_value in zip(feature_names, average_shap_values) if any(shap_value > 0)]
    logging.info("Features with Positive Average SHAP Values:")
    logging.info(f"Total positive features selected: {len(positive_features)} out of total features: {len(feature_names)}, {len(positive_features)}/{len(feature_names)}")
    logging.info(positive_features)
    # print("Total positive features selected:", len(positive_features), "out of total features:", len(feature_names),str(len(positive_features))+"/"+str(len(feature_names)))
    # print(positive_features)
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
    return model,clf

def run(all_csv_file_path, shap_csv_file_path, folder_path, folds):
    logfile = os.path.join(folder_path, 'shap' + '.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(message)s')
    X, y = load_file(all_csv_file_path)
    clf, model = classify(X, y, folds)
    clf_rep(clf, X, y, folds)
    features = shap_feat(model, X)
    X_new, clf2, y = generate_new(features, X, y, folds, shap_csv_file_path)
    clf_rep(clf2, X_new, y, folds)
