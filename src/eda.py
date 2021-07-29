import pandas as pd


def summary_table(df):
    """
    Return a summary table with the descriptive statistics about the dataframe.
    """

    summary = {
    "Number of Variables": [len(df.columns)],
    "Number of Observations": [df.shape[0]],
    "Missing Cells": [df.isnull().sum().sum()],
    "Missing Cells (%)": [round(df.isnull().sum().sum() / df.shape[0] * 100, 2)],
    "Duplicated Rows": [df.duplicated().sum()],
    "Duplicated Rows (%)": [round(df.duplicated().sum() / df.shape[0] * 100, 2)],
    "Categorical Variables": [len([i for i in df.columns if df[i].dtype==object])],
    "Numerical Variables": [len([i for i in df.columns if df[i].dtype!=object])],
    }

    return pd.DataFrame(summary).T.rename(columns={0: 'Values'})