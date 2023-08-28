# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import random

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
import statsmodels.api as sm


# Mute warnings
warnings.filterwarnings("ignore")

import time
RANDOM_STATE = 0


def clean(df):
    # TODO: add "Wd Shng": "Wd Sdng"
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})

    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)

    # Names beginning with numbers are awkward to work with
    df.rename(
        columns={
            "1stFlrSF": "FirstFlrSF",
            "2ndFlrSF": "SecondFlrSF",
            "3SsnPorch": "ThreeSeasonPorch",
        },
        inplace=True,
    )
    return df


# The numeric features are already encoded correctly (`float` for
# continuous, `int` for discrete), but the categoricals we'll need to
# do ourselves. Note in particular, that the `MSSubClass` feature is
# read as an `int` type, but is actually a (nominative) categorical.

# The nominative (unordered) categorical features
features_nom = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
]


# The ordinal (ordered) categorical features

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Add a None level for missing values
ordered_levels = {key: ["None"] + value for key, value in ordered_levels.items()}


def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)

    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels, ordered=True))
    return df


def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df


def load_data():
    # Read data
    df_train = pd.read_csv("train.csv", index_col="Id")
    df_test = pd.read_csv("test.csv", index_col="Id")
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    # Preprocessing
    df = clean(df)
    df = encode(df)
    df = impute(df)
    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return df_train, df_test

df, _ = load_data()

def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    return X


def mathematical_transforms(df):
    X = pd.DataFrame()  # dataframe to hold new features
    # TODO: добавить преобразования
    X["TotalSF"] = df["TotalBsmtSF"] + df["FirstFlrSF"] + df["SecondFlrSF"]

    return X


def interactions(df):
    X = pd.DataFrame()
    # X = pd.get_dummies(df.BldgType, prefix="Bldg")
    # X = X.mul(df.GrLivArea, axis=0)
    return X


def counts(df):
    X = pd.DataFrame()
    #     X["PorchTypes"] = df[[
    #         "WoodDeckSF",
    #         "OpenPorchSF",
    #         "EnclosedPorch",
    #         "ThreeSeasonPorch",
    #         "ScreenPorch",
    #     ]].gt(0.0).sum(axis=1)
    return X


def group_transforms(df):
    X = pd.DataFrame()
    X["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    X["MedNhbdQual"] = df.groupby("Neighborhood")["OverallQual"].transform("median")

    return X


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop("SalePrice")

    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("SalePrice")
        X = pd.concat([X, X_test])

    # Label encoding
    X = label_encode(X)

    # Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    X = X.join(counts(X))
    X = X.join(group_transforms(X))

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    if df_test is not None:
        return X, X_test
    else:
        return X


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    #
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes

    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model,
        X,
        log_y,
        cv=5,
        scoring="neg_mean_squared_error",
    )

    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# xgb_params = dict(
#     max_depth=6,           # maximum depth of each tree - try 2 to 10
#     learning_rate=0.01,    # effect of each tree - try 0.0001 to 0.1
#     n_estimators=1000,     # number of trees (that is, boosting rounds) - try 1000 to 8000
#     min_child_weight=1,    # minimum number of houses in a leaf - try 1 to 10
#     colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
#     subsample=0.7,         # fraction of instances (rows) per tree - try 0.2 to 1.0
#     reg_alpha=0.5,         # L1 regularization (like LASSO) - try 0.0 to 10.0
#     reg_lambda=1.0,        # L2 regularization (like Ridge) - try 0.0 to 10.0
#     num_parallel_tree=1,   # set > 1 for boosted random forests
#     random_state=RANDOM_STATE
# )


def xgb_main(xgb_params, total_runs):
    sum_score = 0
    for i in range(total_runs):
        xgb_params["random_state"] = time.monotonic_ns()
        df_train, df_test = load_data()
        X_train, X_test = create_features(df_train, df_test)
        y_train = df_train.loc[:, "SalePrice"]
        OLS_model = sm.OLS(endog=y_train, exog=sm.add_constant(np.concatenate(
            tuple(
                np.array(X_train[x], dtype=float).reshape(-1, 1)
                for x in X_test.columns
            ),
            axis=1,
        ))).fit()
        datasets = [X_train, X_test]
        for dataset in datasets:
            dataset["OLS"] = OLS_model.predict(exog=sm.add_constant(np.concatenate(
            tuple(
                np.array(dataset[x], dtype=float).reshape(-1, 1)
                for x in X_test.columns
            ),
            axis=1,
        )))

        xgb = XGBRegressor(**xgb_params)
        score = score_dataset(X_train, y_train, model=xgb)
        print(".", end="")
        sum_score += score
    return sum_score / total_runs


def main():
    xgb_params = dict(
        max_depth=6,  # maximum depth of each tree - try 2 to 10
        learning_rate=0.02,  # effect of each tree - try 0.0001 to 0.1
        n_estimators=1000,  # number of trees (that is, boosting rounds) - try 1000 to 8000
        min_child_weight=1,  # minimum number of houses in a leaf - try 1 to 10
        colsample_bytree=0.7,  # fraction of features (columns) per tree - try 0.2 to 1.0
        subsample=0.7,  # fraction of instances (rows) per tree - try 0.2 to 1.0
        reg_alpha=0.5,  # L1 regularization (like LASSO) - try 0.0 to 10.0
        reg_lambda=1.0,  # L2 regularization (like Ridge) - try 0.0 to 10.0
        num_parallel_tree=1,  # set > 1 for boosted random forests
        random_state=RANDOM_STATE,
    )
    params = list(xgb_params.keys())
    params.remove("random_state")
    last_score = 0
    lr = 0.1
    total_runs = 10
    counter = 0
    while True:
        old_xgb_params = {k: v for k, v in xgb_params.items()}  # copy
        if last_score:
            param = random.choice(params)
            if param in ["min_child_weight", "num_parallel_tree", "max_depth"]:
                xgb_params[param] = max(1, xgb_params[param] + random.choice([-1, 1]))
            else:
                change_factor = random.choice([-1, 1]) * lr
                xgb_params[param] += type(xgb_params[param])(xgb_params[param] * change_factor)
        new_score = xgb_main(xgb_params, total_runs)
        if last_score:
            if new_score < last_score:
                print(f" {new_score=}", xgb_params, )
                last_score = new_score
                counter = 0
            else:
                xgb_params = old_xgb_params
                counter += 1
                print(f" Unsuccessful {counter}.")
                if counter == 10:
                    lr /= 1.5
                    counter = 0
        else:
            print(f" base_score={new_score}")
            last_score = new_score
            counter = 0


if __name__ == "__main__":
    main()
