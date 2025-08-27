import os
import pandas as pd


def data_combine(options = ""):
    #Load all raw data files and concatenate them into a single DataFrame, then return the combined DataFrame
    #Combined_df is the df that contains all of the relevant data from all datasets combined
    #options is a string that can be "save" to save the combined df to a csv file

    cols1 = set(pd.read_csv("datasets/2001-02.csv", nrows=0).columns)
    cols2 = set(pd.read_csv("datasets/2021-2022.csv", nrows=0).columns)
    common_cols = sorted(cols1 & cols2)

    print(f"Using {len(common_cols)} common columns:\n{common_cols}\n")

    list_of_files = [
        "datasets/2001-02.csv",
        "datasets/2002-03.csv",
        "datasets/2003-04.csv",
        "datasets/2004-05.csv",
        "datasets/2005-06.csv",
        "datasets/2006-07.csv",
        "datasets/2007-08.csv",
        "datasets/2008-09.csv",
        "datasets/2009-10.csv",
        "datasets/2010-11.csv",
        "datasets/2011-12.csv",
        "datasets/2012-13.csv",
        "datasets/2013-14.csv",
        "datasets/2014-15.csv",
        "datasets/2015-16.csv",
        "datasets/2016-17.csv",
        "datasets/2017-18.csv",
        "datasets/2018-19.csv",
        "datasets/2019-20.csv",
        "datasets/2020-2021.csv",
        "datasets/2021-2022.csv",
        "datasets/2022-2023.csv",
        "datasets/2023-2024.csv",
        "datasets/2024-2025.csv",
        "datasets/2025-2026.csv"
    ]

    dfs = []
    for fname in list_of_files:
        if not os.path.isfile(fname):
            print(f"{fname} not found, skipping")
            continue

        df = pd.read_csv(fname)
        df = df.reindex(columns=common_cols)  # drops extras columns
        dfs.append(df)

    #Concatenate
    combined_df = pd.concat(dfs, ignore_index=True)

    if options == "save":
    #option to save the combined df to a csv file
        combined_df.to_csv("datasets/2001-2025_raw.csv", index=False)
 
    return combined_df
