# %% Imports
from numpy.lib import select
import pandas as pd
import sys
import numpy as np
import random


from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sktools import IsEmptyExtractor
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder
from sklearn.linear_model import QuantileRegressor
from sklego.preprocessing import ColumnSelector 
from sklearn.preprocessing import StandardScaler
from memo import memlist, memfile, grid, time_taken, Runner

sys.path.append("../")

from metrics.metric_participants import (ComputeMetrics, print_metrics)
from eda.checker import check_train_test

random.seed(0)

sales_train = pd.read_csv("../data/data_raw/sales_train.csv")
df_full = pd.read_csv("../data/split.csv")
df_region = pd.read_csv("../data/data_raw/regions.csv")
regions_hcps = pd.read_csv("../data/data_raw/regions_hcps.csv")
activity_features = pd.read_csv("../data/features/activity_features.csv")
brands_3_12 = pd.read_csv("../data/features/brand_3_12_market_features_lagged.csv")
rte_basic = pd.read_csv("../data/features/rte_features_v2.csv").drop(
    columns=["sales", "validation"]
)

market_size = pd.read_csv("../data/market_size.csv")

# For reproducibility
random.seed(0)
VAL_SIZE = 38
SUBMISSION_NAME = "linear_model_simple"

# %% Training weights
market_size = (
    market_size
    .assign(weight=lambda x: 1 / x['sales'])
)
# %% Add region data
df_feats = df_full.merge(df_region, on="region", how="left")
df_feats = pd.merge(left=df_feats, right=regions_hcps, how="left", on="region")
df_feats = df_feats.merge(
    activity_features, on=["month", "region", "brand"], how="left"
)
df_feats = df_feats.merge(rte_basic, on=["month", "region", "brand"], how="left")
df_feats = df_feats.merge(brands_3_12, on=["month", "region"], how="left")
df_feats["whichBrand"] = np.where(df_feats.brand == "brand_1", 1, 0)

df_feats['month_brand'] = df_feats.month + '_' + df_feats.brand

# drop sum variables
cols_to_drop = ["region", "sales", "validation"]

# %% Split train val test
X_train = df_feats.query("validation == 0").drop(columns=cols_to_drop)
y_train = df_feats.query("validation == 0").sales

X_val = df_feats.query("validation == 1").drop(columns=cols_to_drop)
y_val = df_feats.query("validation == 1").sales

X_test = df_feats.query("validation.isnull()", engine="python").drop(
    columns=cols_to_drop
)
y_test = df_feats.query("validation.isnull()", engine="python").sales

check_train_test(X_train, X_val)
check_train_test(X_train, X_test, threshold=0.3)
check_train_test(X_val, X_test)

# %%
for quantile in [0.5, 0.1, 0.9]:

    selected = {}

    for iter in range(100):

        print("Quantile: ", quantile, "iter: ", iter)

        df_train = df_feats.query("validation == 0")

        sample = df_train.sample(replace=True, frac=1)

        X_train = sample.drop(columns=cols_to_drop)
        y_train = sample.sales

        models = {}
        pipes = {}
        train_preds = {}
        val_preds = {}


        models[quantile] = QuantileRegressor(
            quantile=quantile,
            alpha=0.05,
            solver="highs-ds"
        )

        pipes[quantile] = Pipeline(
            [   
                ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
                ("imputer", SimpleImputer(strategy="median")), 
                ("scale", StandardScaler()),
                ("lgb", models[quantile])
            ]
        )

        # Fit cv model
        pipes[quantile].fit(X_train, y_train)
        train_preds[quantile] = pipes[quantile].predict(X_train)
            
        coefs = models[quantile].coef_
        cols_pipe = pipes[quantile][:1].fit_transform(X_train.head(), y_train.head()).columns
        coefs_dict = dict(zip(cols_pipe, coefs))
        selected_features = list({k: v for k, v in coefs_dict.items() if v != 0}.keys())
        selected[iter] = selected_features

    all_selected = {}
    for k, v in selected.items():
        for feature in v:
            all_selected[feature] = all_selected.get(feature, 0) + 1

    all_selected_df = pd.DataFrame(all_selected.items(), columns=["feature", "count"]).sort_values("count", ascending=False)
    all_selected_df.to_csv(f"../data/features/bolasso_features_0{int(quantile * 10)}.csv", index=False)
