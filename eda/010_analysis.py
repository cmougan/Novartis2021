# %%
import pandas as pd

df = pd.read_csv("../data/data_raw/sales_train.csv")
# %%
df.query("month <= '2020-08'").groupby(["month", "brand"]).agg({"sales": ["sum", "max"]})
# %%
df.query("month <= '2020-08'").groupby(["month", "brand"]).agg({"sales": ["sum", "max"]})

# %%
