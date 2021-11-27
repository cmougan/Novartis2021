# %% Imports
import pandas as pd
import sys
import numpy as np

sys.path.append("../")
from metrics.metric_participants import (ComputeMetrics, print_metrics)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklego.preprocessing import ColumnSelector
from sklego.meta import GroupedPredictor, GroupedTransformer
from lightgbm import LGBMRegressor
from category_encoders import TargetEncoder
import random

from eda.checker import check_train_test
from tools.postprocessing import (postprocess_predictions, clip_first_month)
from tools.impute import PandasSimpleImputer

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
SUBMISSION_NAME = "split_gbm_sklego"
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
    'area_x',
    'whichBrand',
]
assert len([col for col in X_train.columns if col in select_cols]) == len(select_cols)

# %%
lgbms = {}
pipes = {}
transforms = {}
train_preds = {}
val_preds = {}
test_preds = {}

for quantile in [0.5, 0.1, 0.9]:

    lgbms[quantile] = GroupedPredictor(
        LGBMRegressor(
            n_jobs=-1,
            n_estimators=50,
            objective="quantile",
            alpha=quantile,
        ),
        groups="whichBrand",
    )

    pipes[quantile] = Pipeline(
        [   
            ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
            ("selector", ColumnSelector(columns=select_cols)),
            ("imputer", PandasSimpleImputer(strategy="median", add_indicator=True)), 
            ("lgb", lgbms[quantile])
        ]
    )

    # Fit cv model
    pipes[quantile].fit(X_train, y_train)
    # , lgb__sample_weight=weights_train)

    train_preds[quantile] = pipes[quantile].predict(X_train)
    val_preds[quantile] = pipes[quantile].predict(X_val)


    if RETRAIN:
        pipes[quantile].fit(X_full, y_full)
        # , lgb__sample_weight=weights_full)
    test_preds[quantile] = pipes[quantile].predict(X_test)


# %% Postprocess
train_preds_post = postprocess_predictions(train_preds)
val_preds_post = postprocess_predictions(val_preds)
test_preds_post = postprocess_predictions(test_preds)

# %% Train prediction
train_preds_post

# %% Train prediction
train_preds_df = (
    df_feats.query("validation == 0")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=train_preds_post[0.5])
    .assign(lower=train_preds_post[0.1])
    .assign(upper=train_preds_post[0.9])
)

ground_truth_train = df_feats.query("validation == 0").loc[
    :, ["month", "region", "brand", "sales"]
]

print_metrics(train_preds_df, sales_train, ground_truth_train)

# %% Validation prediction
val_preds_df = (
    df_feats.query("validation == 1")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=val_preds_post[0.5])
    .assign(lower=val_preds_post[0.1])
    .assign(upper=val_preds_post[0.9])
)

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
