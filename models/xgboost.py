import numpy as np
from util.util import rps, one_hot_y, beta_dist
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

param_grid = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [3, 4, 5, 6, 7, 8],
    "colsample_bytree": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
}

def xgb_no_cv_reg(X_train, y_train, X_val, y_val):
    best_score = 1  
    best_params = None
    best_trees = 0
    best_model = None

    print(f"Testing {len(list(ParameterGrid(param_grid)))} parameter combinations.")
    
    for i, g in enumerate(ParameterGrid(param_grid)):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(list(ParameterGrid(param_grid)))}")
            
        xgb = XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            early_stopping_rounds=50,
            n_estimators=1000,
            random_state=42,
            **g
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        r_train = np.clip(xgb.predict(X_train) / 2.0, 0, 1)
        r_val = np.clip(xgb.predict(X_val) / 2.0, 0, 1)
        probs_val = beta_dist(r_train, y_train, r_val)
        score = rps(probs_val, one_hot_y(y_val))
        
        if score < best_score:
            best_score = score
            best_params = g
            iter = getattr(xgb.get_booster(), "best_iteration", None)
            best_trees = (iter + 1) if iter is not None else xgb.n_estimators
            best_model = xgb
            print(f"New best score: {best_score:.4f}")

    print(f"\nBest parameters: {best_params}")
    print(f"Best RPS score: {best_score:.4f}")
    print(f"Best number of trees: {best_trees}")
    
    return best_model

def predict_with_beta(model, X_train, y_train, X_test):
    r_train = np.clip(model.predict(X_train) / 2.0, 0, 1)
    r_test = np.clip(model.predict(X_test) / 2.0, 0, 1)
    probs_test = beta_dist(r_train, y_train, r_test)
    return probs_test