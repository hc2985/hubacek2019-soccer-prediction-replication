#utility functions
from util.util import preprocess_scaling, rps, one_hot_y, eval
from models.xgboost import xgb_no_cv_reg, predict_with_beta
from modelstorage.modelstorage import savemodel, loadmodel, load_all_models

X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = preprocess_scaling()

xgb_clf = xgb_no_cv_reg(X_train_scaled, y_train, X_val_scaled, y_val)

predictions = predict_with_beta(xgb_clf, X_train_scaled, y_train, X_test_scaled)

y_test_onehot = one_hot_y(y_test)

test_rps = rps(predictions, y_test_onehot)

eval(predictions, y_test)

print(f"Test RPS Score: {test_rps:.4f}")

savemodel({"2017_challenge_winner": xgb_clf})



