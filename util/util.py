from preprocessing.data_combine import data_combine
from preprocessing.data_split import data_split, get_splits
from feature_eng.feature_engineering import featureengineer
from util.metric import rps, one_hot_y
from models.xgboost import xgb_classification
from models.catboost import catboost_classification
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss
import pandas as pd
import numpy as np


def preprocess(file_name = ""):
    #Full process of preprocessing and without scaling
    if file_name == "":
        #Load Data
        combined_df = data_combine()
        #Feature engineering
        feat_df = featureengineer(combined_df, options="save")   
    else:
        feat_df = pd.read_csv(file_name)

    #Split data
    X_train, y_train, X_modern_train, y_modern_train, X_test, y_test = data_split(feat_df)

    return X_train, y_train, X_modern_train, y_modern_train, X_test, y_test

def splits_pipeline(file_name = "", model = "xg"):
    #Full process of preprocessing and without scaling
    if file_name == "":
        #Load Data
        combined_df = data_combine()
        #Feature engineering
        feat_df = featureengineer(combined_df, options="save")   
    else:
        feat_df = pd.read_csv(file_name)

    #store results
    results = []
    #Split data
    splits = get_splits(feat_df)
    for season, X_train, y_train, X_test, y_test in splits:
        y_test_onehot = one_hot_y(y_test)
        if model == "xg":
            clf = xgb_classification(X_train, y_train)
        elif model == "cat":
            clf = catboost_classification(X_train, y_train)
        predictions = clf.predict_proba(X_test) 
        test_rps = rps(predictions, y_test_onehot)
        accuracy, f1, neg_log_loss, brier = eval(predictions, y_test)
        results.append((season, test_rps, accuracy, f1, neg_log_loss, brier))
    return results

def eval(proba, y_test):
    y_pred = np.argmax(proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    neg_log_loss = -log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Neg log Loss: {neg_log_loss:.4f}, brier: {brier:.4f}")
    return accuracy, f1, neg_log_loss, brier