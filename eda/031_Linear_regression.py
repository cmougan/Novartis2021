# %% Imports
import pandas as pd
from doubt import Boot
import sys
from checker import check_train_test

sys.path.append("../")
from metrics.metric_participants import ComputeMetrics
from sklearn.linear_model import LinearRegression
import random

random.seed(0)

sales_train = pd.read_csv("../data/data_raw/sales_train.csv")
df_full = pd.read_csv("../data/split.csv")
df_region = pd.read_csv("../data/data_raw/regions.csv")
regions_hcps = pd.read_csv('../data/data_raw/regions_hcps.csv')

# For reproducibility
random.seed(0)
VAL_SIZE = 38
SUBMISSION_NAME = 'beta'

# %% Add region data
df_feats = df_full.merge(df_region, on="region", how="left")
df_feats = pd.merge(left = df_feats, right=regions_hcps, how='left', on='region')

# drop sum variables
cols_to_drop=['month', 'region', 'brand', 
                'sales','validation']

# %% Split train val test
X_train = df_feats.query("validation == 0").drop(columns=cols_to_drop)
y_train = df_feats.query("validation == 0").sales

X_val = df_feats.query("validation == 1").drop(columns=cols_to_drop)
y_val = df_feats.query("validation == 1").sales

X_test = df_feats.query("validation.isnull()", engine="python").drop(columns=cols_to_drop)
y_test = df_feats.query("validation.isnull()", engine="python").sales

print("Train Validation")
check_train_test(X_train, X_val)
print("Test Train")
check_train_test(X_train, X_test)
print("Test Val")
check_train_test(X_train, X_val)

# %% Train model
model = Boot(LinearRegression(normalize=True))
model.fit(X_train, y_train)

# %% Validation prediction
predictions, intervals = model.predict(X_val, uncertainty=0.2)

val_preds = (
    df_feats.query("validation == 1")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=predictions)
    .assign(lower=intervals[:, 0])
    .assign(upper=intervals[:, 1])
)

ground_truth_val = df_feats.query("validation == 1").loc[
    :, ["month", "region", "brand", "sales"]
]

print(ComputeMetrics(val_preds, sales_train, ground_truth_val))

# %% 
val_preds.to_csv(f'../data/validation/{SUBMISSION_NAME}.csv', index=False)


# %% Test prediction
predictions_test, intervals_test = model.predict(X_test, uncertainty=0.2)
# %%
test_preds = (
    df_feats.query("validation.isnull()", engine="python")
    .loc[:, ["month", "region", "brand"]]
    .assign(sales=predictions_test)
    .assign(lower=intervals_test[:, 0])
    .assign(upper=intervals_test[:, 1])
)

test_preds.to_csv(f'../submissions/{SUBMISSION_NAME}.csv', index=False)

