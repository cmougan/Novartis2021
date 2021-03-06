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
list(X_train.columns)
# %%
select_cols = [
    'whichBrand',
    'count',
    'inverse_tier_f2f',
    'hcp_distinct_Internal medicine / pneumology',
    'hcp_distinct_Internal medicine / pneumology_3m',
    'no. clicks',
    'no. clicks_3m',
    'sales_brand_3',
    'sales_brand_3_market',
    'sales_brand_12_market',
    'month_brand',
    'month',
    'Pediatrician',
    'null_tiers',
]
# select_cols = [
#     'Internal medicine and general practicioner', 'Pediatrician', 'count', 'null_tiers', 'null_tiers_phone', 'inverse_tier_f2f', 'hcp_distinct_Internal medicine / pneumology', 'sales_brand_3', 'sales_brand_12_market', 'month_brand',
# ]

assert len([col for col in X_train.columns if col in select_cols]) == len(select_cols)

# %%
data = []


@memlist(data=data)
def train_and_validate(alpha, X_train, y_train, X_val, df_feats):

    models = {}
    pipes = {}
    train_preds = {}
    val_preds = {}

    for quantile in [0.5, 0.1, 0.9]:

        models[quantile] = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver="highs-ds"
        )

        pipes[quantile] = Pipeline(
            [   
                ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)), 
                ("scale", StandardScaler()),
                ("lgb", models[quantile])
            ]
        )

        # Fit cv model
        pipes[quantile].fit(X_train, y_train)

        train_preds[quantile] = pipes[quantile].predict(X_train)
        val_preds[quantile] = pipes[quantile].predict(X_val)

    # %% Validation prediction
    val_preds_df = (
        df_feats.query("validation == 1")
        .loc[:, ["month", "region", "brand"]]
        .assign(sales=val_preds[0.5].clip(0))
        .assign(lower=val_preds[0.1].clip(0))
        .assign(upper=val_preds[0.9].clip(0))
    )

    ground_truth_val = df_feats.query("validation == 1").loc[
        :, ["month", "region", "brand", "sales"]
    ]
    metrics = ComputeMetrics(val_preds_df, sales_train, ground_truth_val)

    train_preds_df = (
        df_feats.query("validation == 0")
        .loc[:, ["month", "region", "brand"]]
        .assign(sales=train_preds[0.5].clip(0))
        .assign(lower=train_preds[0.1].clip(0))
        .assign(upper=train_preds[0.9].clip(0))
    )

    ground_truth_train = df_feats.query("validation == 0").loc[
        :, ["month", "region", "brand", "sales"]
    ]

    metrics_train = ComputeMetrics(train_preds_df, sales_train, ground_truth_train)
    return {"accuracy": metrics[0], "deviation": metrics[1], "accuracy_train": metrics_train [0], "deviation_train": metrics_train[1]}

# %%
partial_train_and_validate = partial(
    train_and_validate, 
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    df_feats=df_feats
)

settings = grid(alpha=[0, 0.0005, 0.001, 0.005])

# To Run in parallel
runner = Runner()
runner.run(
    func=partial_train_and_validate,
    settings=settings, 
)

# %%
for elem in data:
    print(elem['alpha'])
    print(elem['accuracy'])
    print(elem['accuracy_train'])
    print(elem['deviation'])
    print(elem['deviation_train'])
# %%


models = {}
pipes = {}
train_preds = {}
val_preds = {}

for quantile in [0.5]:

    models[quantile] = QuantileRegressor(
        quantile=quantile,
        alpha=0.05,
        solver="highs-ds"
    )

    pipes[quantile] = Pipeline(
        [   
            ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
            ("empty", IsEmptyExtractor(cols=["count", "count_other", "inverse_tier_other", "count_Pediatrician"])),
            # ("selector", ColumnSelector(columns=select_cols)),
            ("imputer", SimpleImputer(strategy="median")), 
            ("scale", StandardScaler()),
            ("lgb", models[quantile])
        ]
    )

    # Fit cv model
    pipes[quantile].fit(X_train, y_train)

    train_preds[quantile] = pipes[quantile].predict(X_train)
    val_preds[quantile] = pipes[quantile].predict(X_val)
# %%
coefs = models[0.5].coef_

# %%
cols_pipe = pipes[0.5][:2].fit_transform(X_train.head(), y_train.head()).columns

# %%
# Join two lists into dictionary
coefs_dict = dict(zip(cols_pipe, coefs))
# %%
{k: v for k, v in coefs_dict.items() if v != 0}.keys()
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
    'brand'
]

# %%
models = {}
pipes = {}
train_preds = {}
val_preds = {}
test_preds = {}

for quantile in [0.5, 0.1, 0.9]:

    print("Quantile:", quantile)
    models[quantile] = QuantileRegressor(
        quantile=quantile,
        alpha=0,
        solver="highs-ds"
    )

    pipes[quantile] = Pipeline(
        [   
            ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
            ("selector", ColumnSelector(columns=select_cols)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)), 
            ("scale", StandardScaler()),
            ("qr", models[quantile])
        ]
    )

    # Fit cv model
    pipes[quantile].fit(X_train, y_train)
    # , qr__sample_weight=weights_train)

    train_preds[quantile] = pipes[quantile].predict(X_train)
    val_preds[quantile] = pipes[quantile].predict(X_val)
# %%
data = []


# %%
@memlist(data=data)
def evaluate_val_preds(alpha):
    models = {}
    pipes = {}
    train_preds = {}
    val_preds = {}
    test_preds = {}

    lower = alpha / 2
    upper = 1 - (alpha / 2)
    for quantile in [upper, lower]:

        print("Quantile:", quantile)
        models[quantile] = QuantileRegressor(
            quantile=quantile,
            alpha=0,
            solver="highs-ds"
        )

        pipes[quantile] = Pipeline(
            [   
                ("te", TargetEncoder(cols=["month_brand", "month", "brand"])),
                ("selector", ColumnSelector(columns=select_cols)),
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)), 
                ("scale", StandardScaler()),
                ("qr", models[quantile])
            ]
        )

        # Fit cv model
        pipes[quantile].fit(X_train, y_train)
        # , qr__sample_weight=weights_train)

        train_preds[quantile] = pipes[quantile].predict(X_train)
        val_preds[quantile] = pipes[quantile].predict(X_val)

    # %% Validation prediction
    val_preds_df = (
        df_feats.query("validation == 1")
        .loc[:, ["month", "region", "brand"]]
        .assign(sales=val_preds[upper].clip(0))
        .assign(lower=val_preds[lower].clip(0))
        .assign(upper=val_preds[upper].clip(0))
    )

    ground_truth_val = df_feats.query("validation == 1").loc[
        :, ["month", "region", "brand", "sales"]
    ]
    metrics = ComputeMetrics(val_preds_df, sales_train, ground_truth_val)

    return {"deviation": metrics[1]}

# %%

settings = grid(alpha=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

# To Run in parallel
runner = Runner()
runner.run(
    func=evaluate_val_preds,
    settings=settings, 
)

# %%
for elem in data:
    print(elem['alpha'])
    print(elem['deviation'])
# %%
