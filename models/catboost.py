from catboost import CatBoostClassifier, CatBoostRegressor
from util.util import rps, one_hot_y, beta_dist

def catboost_no_cv(X_train, y_train, X_val, y_val):
    cat = CatBoostRegressor()