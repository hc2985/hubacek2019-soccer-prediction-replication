from sklearn.preprocessing import StandardScaler
import pandas as pd

def data_scale(X_train, X_modern_train, X_test):
    # Scale the data using StandardScaler
    cols = X_train.columns
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=cols)
    X_modern_scaled = pd.DataFrame(scaler.transform(X_modern_train), columns=cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=cols)

    return X_train_scaled, X_modern_scaled, X_test_scaled