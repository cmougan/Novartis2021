# %% Imports
import pandas as pd
import sys
import numpy as np

sys.path.append("../")
from metrics.metric_participants import (ComputeMetrics, print_metrics)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklego.preprocessing import ColumnSelector
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder
import random

from eda.checker import check_train_test
from tools.postprocessing import (
    clip_first_month, postprocess_submissions
)

random.seed(0)

sales_train = pd.read_csv("../data/data_raw/sales_train.csv")
df_full = pd.read_csv("../data/split.csv")
df_region = pd.read_csv("../data/data_raw/regions.csv")
regions_hcps = pd.read_csv("../data/data_raw/regions_hcps.csv")
activity_features = pd.read_csv("../data/features/activity_features.csv")
brands_3_12 = pd.read_csv("../data/features/brand_3_12_market_features_lagged.csv")
rte_basic = pd.read_csv("../data/features/rte_basic_features.csv").drop(
    columns=["sales", "validation"]
)

market_size = pd.read_csv("../data/market_size.csv")

# For reproducibility
random.seed(0)
VAL_SIZE = 38
SUBMISSION_NAME = "split_gbm"
RETRAIN = True

# %% Training weights
market_size = (
    market_size
    .assign(weight=lambda x: 100 / x['sales'])
    .rename(columns={"sales": 'market_size'})
)

market_size

# %% Add region data
df_feats = df_full.merge(df_region, on="region", how="left")
df_feats = pd.merge(left=df_feats, right=regions_hcps, how="left", on="region")
df_feats = df_feats.merge(
    activity_features, on=["month", "region", "brand"], how="left"
)
df_feats = df_feats.merge(rte_basic, on=["month", "region", "brand"], how="left")
df_feats = df_feats.merge(brands_3_12, on=["month", "region"], how="left")
df_feats["whichBrand"] = np.where(df_feats.brand == "brand_1", 1, 0)

df_feats = df_feats.merge(market_size, on='region', how="left")

df_feats['month_brand'] = df_feats.month + '_' + df_feats.brand

# drop sum variables
cols_to_drop = ["region", "sales", "validation", "market_size", "weight"]

# %% Split train val test
X_train = df_feats.query("validation == 0").drop(columns=cols_to_drop)
y_train = df_feats.query("validation == 0").sales
weights_train = df_feats.query("validation == 0").weight

X_val = df_feats.query("validation == 1").drop(columns=cols_to_drop)
y_val = df_feats.query("validation == 1").sales

X_full = df_feats.query("validation.notnull()", engine="python").drop(
    columns=cols_to_drop
)
y_full = df_feats.query("validation.notnull()", engine="python").sales
weights_full = df_feats.query("validation.notnull()", engine="python").weight

X_test = df_feats.query("validation.isnull()", engine="python").drop(
    columns=cols_to_drop
)
y_test = df_feats.query("validation.isnull()", engine="python").sales

check_train_test(X_train, X_val)
check_train_test(X_train, X_test, threshold=0.3)
check_train_test(X_val, X_test)

# %%
select_cols = [
    'whichBrand',
    'count',
    'inverse_tier_f2f',
    'hcp_distinct_Internal medicine / pneumology',
    'sales_brand_3',
    'sales_brand_3_market',
    'sales_brand_12_market',
    'month_brand',
    'month',
]


select_cols = [
    "month_brand",
    "sales_brand_3",
    "inverse_tier_f2f",
    "hcp_distinct_Internal medicine / pneumology",
    "sales_brand_12_market_per_region",
    "sales_brand_12_market",
    'no. openings_Pediatrician',
    'tier_openings_Internal medicine / pneumology',
    'area_x'
]
assert len([col for col in X_train.columns if col in select_cols]) == len(select_cols)

# %%
brands = X_train.brand.sort_values().unique()

Xs = {}
ys = {}
Xs_val = {}
ys_val = {}
Xs_test = {}
ys_test = {}
Xs_full = {}
ys_full = {}

lgbms = {}
pipes = {}
train_preds = {}
val_preds = {}
test_preds = {}

for i, brand in enumerate(list(brands)):
    
    Xs[brand] = X_train.copy()[X_train.brand == brand]
    ys[brand] = y_train.copy()[X_train.brand == brand]
    Xs_val[brand] = X_val.copy()[X_val.brand == brand]
    ys_val[brand] = y_val.copy()[X_val.brand == brand]
    Xs_test[brand] = X_test.copy()[X_test.brand == brand]
    ys_test[brand] = y_test.copy()[X_test.brand == brand]
    Xs_full[brand] = X_full.copy()[X_full.brand == brand]
    ys_full[brand] = y_full.copy()[X_full.brand == brand]
    lgbms[brand] = {}
    pipes[brand] = {}
    train_preds[brand] = {}
    val_preds[brand] = {}
    test_preds[brand] = {}

    for quantile in [0.5, 0.1, 0.9]:

        lgbms[brand][quantile] = LGBMRegressor(
            n_jobs=-1,
            n_estimators=25,
            objective="quantile",
            alpha=quantile,
        )

        pipes[brand][quantile] = Pipeline(
            [   
                ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
                ("selector", ColumnSelector(columns=select_cols)),
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)), 
                ("lgb", lgbms[brand][quantile])
            ]
        )

        # Fit cv model
        pipes[brand][quantile].fit(Xs[brand], ys[brand])

        train_preds[brand][quantile] = pipes[brand][quantile].predict(Xs[brand])
        val_preds[brand][quantile] = pipes[brand][quantile].predict(Xs_val[brand])

        if RETRAIN:
            pipes[brand][quantile].fit(Xs_full[brand], ys_full[brand])
            # , qr__sample_weight=weights_full)
        test_preds[brand][quantile] = pipes[brand][quantile].predict(Xs_test[brand])


# %% Postprocess

name_mapping = {"sales": 0.5, "lower": 0.1, "upper": 0.9}

train_preds_df = (
    df_feats.query("validation == 0")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=0)
    .assign(lower=0)
    .assign(upper=0)
)

for brand, d in train_preds.items():
    for name, quantile in name_mapping.items():
        train_preds_df.loc[train_preds_df.brand == brand, name] = d[quantile]

val_preds_df = (
    df_feats.query("validation == 1")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=0)
    .assign(lower=0)
    .assign(upper=0)
)

for brand, d in val_preds.items():
    for name, quantile in name_mapping.items():
        val_preds_df.loc[val_preds_df.brand == brand, name] = d[quantile]

test_preds_df = (
    df_feats.query("validation.isnull()", engine="python")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=0)
    .assign(lower=0)
    .assign(upper=0)
)

for brand, d in test_preds.items():
    for name, quantile in name_mapping.items():
        test_preds_df.loc[test_preds_df.brand == brand, name] = d[quantile]


# %% Postprocessing
train_preds_post = clip_first_month(postprocess_submissions(train_preds_df))
val_preds_post = clip_first_month(postprocess_submissions(val_preds_df))
test_preds_post = clip_first_month(postprocess_submissions(test_preds_df))

# %%
ground_truth_train = df_feats.query("validation == 0").loc[
    :, ["month", "region", "brand", "sales"]
]

print_metrics(train_preds_df, sales_train, ground_truth_train)


ground_truth_val = df_feats.query("validation == 1").loc[
    :, ["month", "region", "brand", "sales"]
]

print_metrics(val_preds_df, sales_train, ground_truth_val)

# %%
val_preds_df.to_csv(f"../data/validation/{SUBMISSION_NAME}_val.csv", index=False)


# %% Test prediction
test_preds_df = (
    df_feats.query("validation.isnull()", engine="python")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=test_preds_post[0.5])
    .assign(lower=test_preds_post[0.1])
    .assign(upper=test_preds_post[0.9])
)

test_preds_df.to_csv(f"../submissions/{SUBMISSION_NAME}.csv", index=False)
