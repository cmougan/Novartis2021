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

# For reproducibility
random.seed(0)
VAL_SIZE = 38
SUBMISSION_NAME = "beta"

# %% Add region data
df_feats = df_full.merge(df_region, on="region", how="left")

# %% Split train val test
X_train = df_feats.query("validation == 0").loc[:, ["area", "population"]]
y_train = df_feats.query("validation == 0").sales

X_val = df_feats.query("validation == 1").loc[:, ["area", "population"]]
y_val = df_feats.query("validation == 1").sales

X_test = df_feats.query("validation.isnull()", engine="python").loc[
    :, ["area", "population"]
]
y_test = df_feats.query("validation.isnull()", engine="python").sales

print("Train Validation")
check_train_test(X_train, X_val)
print("Test Train")
check_train_test(X_train, X_test)
print("Test Val")
check_train_test(X_train, X_val)

# %% Train model
model = Boot(LinearRegression())
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

test_preds.to_csv(f"../submissions/{SUBMISSION_NAME}.csv", index=False)
