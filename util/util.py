from preprocessing.data_combine import data_combine
from preprocessing.data_split import data_split
from preprocessing.data_scale import data_scale
from feature_eng.feature_engineering import featureengineer
from scipy.stats import beta
from sklearn.metrics import accuracy_score, f1_score, log_loss, brier_score_loss
import pandas as pd
import numpy as np


def preprocess_scaling(file_name = ""):
    #Full process of preprocessing and scaling

    if file_name == "":
        #Load Data
        combined_df = data_combine()
        #Feature engineering
        feat_df = featureengineer(combined_df, options="save")   
    else:
        feat_df = pd.read_csv(file_name)

    #Split data
    X_train, y_train, X_modern_train, y_modern_train, X_test, y_test = data_split(feat_df)
    #Scale data
    X_train_scaled, X_modern_scaled, X_test_scaled = data_scale(X_train, X_modern_train, X_test)

    return X_train_scaled, y_train, X_modern_scaled, y_modern_train, X_test_scaled, y_test

def rps(probs, outcome_onehot):
    probs = np.asarray(probs)
    outcome_onehot = np.asarray(outcome_onehot)
    if probs.ndim == 1: 
        probs = probs.reshape(outcome_onehot.shape[0], -1)
    return np.mean(np.sum((np.cumsum(probs, axis=1) - np.cumsum(outcome_onehot, axis=1))**2, axis=1) / (probs.shape[1]-1))

def one_hot_y(y):
    out = np.zeros((len(y), 3), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out

def beta_dist(r_train, y_train, r_val, eps=1e-6):
    r_train = np.clip(np.asarray(r_train).ravel(), eps, 1.0 - eps)
    r_val   = np.clip(np.asarray(r_val).ravel(),   eps, 1.0 - eps)
    y_train = np.asarray(y_train).ravel()

    probs_val = np.zeros((len(r_val), 3), dtype=float)

    for k in (0, 1, 2):
        vals = r_train[y_train == k]
        a, b, loc, scale = beta.fit(vals, floc=0.0, fscale=1.0)           
        probs_val[:, k] = beta.pdf(r_val, a, b, loc=0.0, scale=1.0)

    probs_val = np.clip(probs_val, eps, None)
    row_sums = probs_val.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs_val /= row_sums
    
    return probs_val

def rps_scorer(estimator, X, y):
    # Get regression predictions by scaling to beta distribution
    r_pred = estimator.predict(X)
    r_pred = np.clip(r_pred / 2.0, 0, 1) 
    probs = beta_dist(r_pred, y, r_pred)
    score = rps(probs, one_hot_y(y))
    return -score

def eval(proba, y_test):
    y_pred = np.argmax(proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    neg_log_loss = -log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Neg log Loss: {neg_log_loss:.4f}, brier: {brier:.4f}")