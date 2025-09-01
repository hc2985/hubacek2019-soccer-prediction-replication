#utility functions
from util.util import splits_pipeline
import joblib
import os

results = splits_pipeline()

for season, test_rps, accuracy, f1, neg_log_loss, brier in results:
    print(f"Season: {season}, Test RPS Score: {test_rps:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Neg Log Loss: {neg_log_loss:.4f}, Brier Score: {brier:.4f}")


joblib.dump(results, os.path.join("modelstorage", f"results_original.joblib"))

