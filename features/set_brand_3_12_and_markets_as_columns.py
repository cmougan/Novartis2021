
# %% Imports
import pandas as pd

sales_train = pd.read_csv("../data/data_raw/sales_train.csv")
df_full = pd.read_csv("../data/split.csv")

# %% All
sales_train_brand3 = sales_train[sales_train['brand']=='brand_3'].drop(columns='brand').rename(columns={'sales':'sales_brand_3'})
sales_train_brand3_m = sales_train[sales_train['brand']=='brand_3_market'].drop(columns='brand').rename(columns={'sales':'sales_brand_3_market'})
sales_train_brand12_m = sales_train[sales_train['brand']=='brand_12_market'].drop(columns='brand').rename(columns={'sales':'sales_brand_12_market'})

pd.merge(sales_train_brand3,sales_train_brand3_m, on=['month','region'])

# %%
