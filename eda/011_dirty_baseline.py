# %% Imports
import random
import pandas as pd
from doubt import Boot
import sys
sys.path.append('../')
from metrics.metric_participants import ComputeMetrics
from sklearn.linear_model import LinearRegression

sales_train = pd.read_csv('../data/data_raw/sales_train.csv')
df_submission = pd.read_csv('../data/data_raw/submission_sample.csv')
df_region = pd.read_csv('../data/data_raw/regions.csv')

# For reproducibility
random.seed(0)
VAL_SIZE = 38

# %% Keep only brands 1 and 2
brands_12 = sales_train[sales_train.brand.isin(['brand_1', 'brand_2'])]
brands_12_train = brands_12[brands_12.month >= '2020-07'].reset_index(drop=True)

# %% Validation data
val_regions = pd.Series(brands_12_train.region.unique()).sample(n=VAL_SIZE).reset_index(drop=True)
validation_ids = pd.DataFrame(dict(region=val_regions, validation=[1] * VAL_SIZE))

# %% Add validation column
brands_12_train = (
    brands_12_train
    .merge(validation_ids, on='region', how='left')
    .assign(validation=lambda x: x.validation.fillna(0))
)

# %% Create test data
df_test = (
    df_submission
    .loc[:, ["month", 'region', 'brand']]
    .assign(sales=None)
    .assign(validation=None)
)

# %% Train + test data
df_full = pd.concat([brands_12_train, df_test], sort=False)
df_full

# %% Add region data
df_feats = df_full.merge(df_region, on='region', how='left')

# %% Split train val test
X_train = df_feats.query('validation == 0').loc[:, ['area', 'population']]
y_train = df_feats.query('validation == 0').sales

X_val = df_feats.query('validation == 1').loc[:, ['area', 'population']]
y_val = df_feats.query('validation == 1').sales

X_test = df_feats.query('validation.isnull()', engine='python').loc[:, ['area', 'population']]
y_test = df_feats.query('validation.isnull()', engine='python').sales


# %% Train model
model = Boot(LinearRegression())
model.fit(X_train, y_train)

# %% Validation prediction
predictions, intervals = model.predict(X_val, uncertainty=0.2)

val_preds = (
    df_feats
    .query('validation == 1')
    .loc[:, ['month', 'region', 'brand']]
    .assign(sales=predictions)
    .assign(lower=intervals[:, 0])
    .assign(upper=intervals[:, 1])
)

ground_truth_val = (
    df_feats
    .query('validation == 1')
    .loc[:, ['month', 'region', 'brand', 'sales']]
)

ComputeMetrics(val_preds, sales_train, ground_truth_val)
# %% Test prediction
predictions_test, intervals_test = model.predict(X_test, uncertainty=0.2)
# %%
test_preds = (
    df_feats
    .query('validation.isnull()', engine='python')
    .loc[:, ['month', 'region', 'brand']]
    .assign(sales=predictions_test)
    .assign(lower=intervals_test[:, 0])
    .assign(upper=intervals_test[:, 1])
)

test_preds.to_csv('../submissions/beta.csv')
