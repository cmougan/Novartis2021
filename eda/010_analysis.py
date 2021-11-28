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
outside = pd.read_csv("../data/metric_outside.csv")
width = pd.read_csv("../data/metric_width.csv")
# %%
errors = outside.merge(width, on=["region", "brand"]).assign(error=lambda x: x["outside"] + x["width"])
# %%
errors.sum()
# %%
errors.sort_values("error", ascending=False).head(10)
# %%
errors.sort_values("outside", ascending=False).head(10)

# %%
errors.sort_values("width", ascending=False).head(10)

# %%
