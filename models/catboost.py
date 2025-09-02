import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from util.util import rps, one_hot_y

param_grid = {
    "depth": [3, 4, 5, 6, 7, 8],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_data_in_leaf": [1, 2, 3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1]
}


def catboost_classification(X_train, y_train, n_splits=5, iterations=1000, 
                           early_stopping_rounds=10, random_state=42):
    
    # Base parameters for CatBoost classification
    params_base = {
        "objective": "MultiClass",
        "eval_metric": "MultiClass",  # This uses logloss internally
        "task_type": "GPU",
        "devices": "0",  # Use GPU 0
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "bootstrap_type": "Poisson",
    }

    tscv = TimeSeriesSplit(n_splits=n_splits)

    best = None  
    
    # Calculate total parameter combinations for progress tracking
    param_combinations = list(ParameterGrid(param_grid))
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

            # Create CatBoost pools
            train_pool = Pool(X_tr, label=y_tr)
            val_pool = Pool(X_val, label=y_val)

            # Create and train CatBoost model
            model = CatBoostClassifier(
                **params_base,
                **p,
                iterations=iterations,
                early_stopping_rounds=early_stopping_rounds
            )
            
            # Suppress output during training
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    verbose=False,
                    plot=False
                )

            # Get probability predictions for validation set
            probs_val = model.predict_proba(X_val)
            onehot_val = one_hot_y(y_val)
            
            # Calculate RPS for this fold
            fold_rps.append(rps(probs_val, onehot_val))
            fold_rounds.append(model.get_best_iteration() + 1)

        mean_rps = float(np.mean(fold_rps))
        chosen_round = int(np.median(fold_rounds))

        if best is None or mean_rps < best[0]:
            best = (mean_rps, chosen_round, p, fold_rounds)

    mean_rps, chosen_round, best_params, _ = best

    # Refit on ALL training data with chosen params/rounds
    print(f"\nTraining final model with best params: {best_params}")
    print(f"Using {chosen_round} rounds...")
    
    full_pool = Pool(X_train, label=y_train)
    
    final_model = CatBoostClassifier(
        **params_base,
        **best_params,
        iterations=chosen_round,
        early_stopping_rounds=None  # No early stopping for final model
    )
    
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        final_model.fit(full_pool, verbose=False, plot=False)
    
    print("Final model training completed!")

    return CatBoostClassifierWrapper(
        model=final_model,
        best_params=best_params,
        best_round=chosen_round,
        best_cv_rps=mean_rps
    )

class CatBoostClassifierWrapper:
    """Wrapper class to maintain consistency with XGBoost implementation"""
    
    def __init__(self, model, best_params, best_round, best_cv_rps):
        self.model = model
        self.best_params = best_params
        self.best_round = best_round
        self.best_cv_rps = best_cv_rps
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Get class predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self, importance_type='PredictionValuesChange'):
        """Get feature importance"""
        return self.model.get_feature_importance(type=importance_type)
    
    def save_model(self, filename):
        """Save the model"""
        self.model.save_model(filename)
    
    @classmethod
    def load_model(cls, filename):
        """Load a saved model"""
        model = CatBoostClassifier()
        model.load_model(filename)
        return cls(model=model, best_params={}, best_round=0, best_cv_rps=0.0)