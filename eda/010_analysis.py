# %%
import pandas as pd

df = pd.read_csv('../data/data_raw/sales_train.csv')
# %%
df.groupby(['month', 'brand']).agg({'sales': ['sum', 'max']})
# %%
