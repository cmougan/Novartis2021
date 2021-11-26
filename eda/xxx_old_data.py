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
brands_12_train = brands_12[brands_12.month < "2020-07"].reset_index(drop=True)
# %%
brands_12_train.sales.describe()
# %%
