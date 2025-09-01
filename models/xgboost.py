import numpy as np
from sklearn.model_selection import ParameterGrid, GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

param_grid = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [3, 4, 5, 6, 7, 8],
    "colsample_bytree": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
}

def xgb_cv(X_train, y_train):
    cv = TimeSeriesSplit(n_splits=5)
    model = XGBRegressor(device = "cuda", early_stopping_rounds=30, random_state=42)
    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='neg_mean_squared_error', verbose=1, n_jobs=-3)
    clf.fit(X_train, y_train)
    return clf.best_estimator_