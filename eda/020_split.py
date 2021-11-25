# %% Imports
import random
import pandas as pd

sales_train = pd.read_csv("../data/data_raw/sales_train.csv")
df_submission = pd.read_csv("../data/data_raw/submission_sample.csv")

# For reproducibility
random.seed(0)
VAL_SIZE = 38

# %% Keep only brands 1 and 2
brands_12 = sales_train[sales_train.brand.isin(["brand_1", "brand_2"])]
brands_12_train = brands_12[brands_12.month >= "2020-07"].reset_index(drop=True)

# %% Validation data
val_regions = (
    pd.Series(brands_12_train.region.unique()).sample(n=VAL_SIZE).reset_index(drop=True)
)
validation_ids = pd.DataFrame(dict(region=val_regions, validation=[1] * VAL_SIZE))

# %% Add validation column
brands_12_train = brands_12_train.merge(validation_ids, on="region", how="left").assign(
    validation=lambda x: x.validation.fillna(0)
)

# %% Create test data
df_test = (
    df_submission.loc[:, ["month", "region", "brand"]]
    .assign(sales=None)
    .assign(validation=None)
)

# %% Train + test data
df_full = pd.concat([brands_12_train, df_test], sort=False)
df_full

# %%
df_full.to_csv("../data/split.csv", index=False)
# %%
