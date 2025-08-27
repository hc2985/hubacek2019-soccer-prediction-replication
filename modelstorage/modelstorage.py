import os
import joblib

def savemodel(models):
    #stores trained models for future use
    for tag, model in models.items():
        model_name = type(model).__name__
        joblib.dump(model, os.path.join("modelstorage", f"{model_name}_{tag}.joblib"))

def loadmodel(model_name, tag=""):
    #loads trained models
    return joblib.load(os.path.join("modelstorage", f"{model_name}_{tag}.joblib"))

def load_all_models(folder="modelstorage"):
    #load all models in modelstorage without specifying name
    models = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".joblib"):
            continue
        stem = fname[:-7] 
        model_name, tag = stem.split("_", 1)   
        models[stem] = loadmodel(model_name, tag)
        

    print(f"Loaded {len(models)} models from {folder}")
    return models