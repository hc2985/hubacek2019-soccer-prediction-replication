import numpy as np
from scipy.stats import beta

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