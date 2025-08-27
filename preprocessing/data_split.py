import pandas as pd

def data_split(df, use_case="train"):
    #split data into train, validation and test sets based on date
    if use_case == "train":
        date = pd.to_datetime(df['Date'], dayfirst=False, yearfirst=True, errors='coerce')
        cols = df.columns.drop(['Date','Home_Team','Away_Team', 'Match_Result'])

        mask_train = (date >= '2000-08-01') & (date < '2018 -08-01')
        mask_val =  (date >= '2018-08-01') & (date < '2023-08-01')
        mask_test = (date >= '2023-08-01') & (date < '2025-08-01')

        train_df = df.loc[mask_train]
        val_df = df.loc[mask_val]
        test_df = df.loc[mask_test]

        y_train = train_df.pop('Match_Result').values
        X_train = train_df.drop(columns=['Date','Home_Team','Away_Team']).values

        y_val = val_df.pop('Match_Result').values
        X_val = val_df.drop(columns=['Date','Home_Team','Away_Team']).values

        y_test = test_df.pop('Match_Result').values
        X_test = test_df.drop(columns=['Date','Home_Team','Away_Team']).values

        X_train = pd.DataFrame(X_train, columns=cols)
        X_val = pd.DataFrame(X_val, columns=cols)
        X_test = pd.DataFrame(X_test, columns=cols)

        return X_train, y_train, X_val, y_val, X_test, y_test

    elif use_case == "display_test":
        date = pd.to_datetime(df['Date'], dayfirst=True, yearfirst=False, errors='coerce')
        mask_test = date >= '2025-08-01'
        display_df = df.loc[mask_test]
        X_display = pd.DataFrame(display_df)

        return X_display
