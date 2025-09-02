import itertools
import numpy as np
import xgboost as xgb
from util.metric import rps, one_hot_y
from sklearn.model_selection import TimeSeriesSplit
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO


param_grid = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [3, 4, 5, 6, 7, 8],
    "colsample_bytree": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
}

def _grid_iter(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def _predict_at_best(booster, dmat):
    bi = getattr(booster, "best_iteration", None)
    if isinstance(bi, int) and bi >= 0:
        return booster.predict(dmat, iteration_range=(0, bi + 1))
    #fallback: use all trees
    return booster.predict(dmat)

class XGBClassifierWrapper:
    def __init__(self, booster, best_params, best_round, best_cv_rps):
        self.booster = booster
        self.best_params_ = best_params
        self.best_iteration_ = best_round
        self.best_score_ = best_cv_rps  
        self.n_classes_ = 3

    def predict(self, X):
        """Returns class predictions (0=loss, 1=draw, 2=win)"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """Returns probability matrix with shape (n_samples, 3)"""
        dmat = xgb.DMatrix(X)
        return _predict_at_best(self.booster, dmat)

def xgb_classification(X_train, y_train, n_splits=5, num_boost_round=1000, 
                      early_stopping_rounds=10, random_state=42):
    
    #base parameters for XGBoost classification
    params_base = {
        "objective": "multi:softprob",  
        "num_class": 3,                 
        "device": "cuda",
        "seed": random_state,
        "eval_metric": "mlogloss",      
        "max_bin": 512,                 
        "grow_policy": "depthwise",     
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    best = None  
    
    #calculate total parameter combinations for progress tracking
    param_combinations = list(_grid_iter(param_grid))
    total_combinations = len(param_combinations)
    
    print(f"Starting grid search with {total_combinations} parameter combinations...")

    for combo_idx, p in enumerate(param_combinations, 1):
        fold_rps = []
        fold_rounds = []

        if combo_idx % 100 == 0: 
            print(f"\nParameter combination {combo_idx}/{total_combinations}")

        for tr_idx, val_idx in tscv.split(X_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            booster = xgb.train(
                params={**params_base, **p},
                dtrain=dtr,
                num_boost_round=num_boost_round,
                evals=[(dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0
            )

            #get probability predictions for validation set
            probs_val = _predict_at_best(booster, dval)
            onehot_val = one_hot_y(y_val)
            
            #calculate RPS for this fold
            fold_rps.append(rps(probs_val, onehot_val))
            fold_rounds.append(booster.best_iteration + 1)

        mean_rps = float(np.mean(fold_rps))
        chosen_round = int(np.median(fold_rounds))

        if best is None or mean_rps < best[0]:
            best = (mean_rps, chosen_round, p, fold_rounds)

    mean_rps, chosen_round, best_params, _ = best

    #refit on ALL training data with chosen params/rounds
    print(f"\nTraining final model with best params: {best_params}")
    print(f"Using {chosen_round} rounds...")
    
    dfull = xgb.DMatrix(X_train, label=y_train)
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        final_booster = xgb.train(
            params={**params_base, **best_params},
            dtrain=dfull,
            num_boost_round=chosen_round,
            verbose_eval=False,
        )
    
    print("Final model training completed!")

    return XGBClassifierWrapper(
        booster=final_booster,
        best_params=best_params,
        best_round=chosen_round,
        best_cv_rps=mean_rps
    )