# %%
import pandas as pd
import altair as alt

df = pd.read_csv("../data/data_raw/sales_train.csv")

valid_gbm = pd.read_csv("../data/validation/empty_extractor_target_encoder_val.csv").rename(columns={"sales": "pred"})
# %%

df_valid = df.merge(valid_gbm, on=["month", "region", "brand"])
# %%
df_valid
# %%
region_example = df_valid.query("region == 'region_134'")

# %%
p1 = alt.Chart(
    region_example
).mark_line().encode(
    x="month",
    y='sales',
    color='brand',
)
# .mark_line().encode(
#     x="month",
#     y='pred',
#     color='black'
# )
# %%
p2 = alt.Chart(
    region_example
).mark_line().encode(
    x="month",
    y='pred',
    color='brand'
)
# %%
p1 + p2
# %%
