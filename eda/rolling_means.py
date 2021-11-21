# %% Load birthday data
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from tools.rolling import rolling_fn

df = pd.read_csv("https://calmcode.io/datasets/birthdays.csv")

# %% Create a subset with only data from CA, parse date properly
subset_df = (
    df[['state', 'date', 'births']]
    .assign(date=lambda d: pd.to_datetime(d['date'], format="%Y-%m-%d"))
    .loc[lambda d: d['state'] == 'CA']
    .tail(365 * 2)
)


# %% Without using series
subset_df.assign(
    rolling_births=lambda d: (
        d
        # Do not use current row, only previous ones
        .shift(1)
        # Do rolling average of 10 rows, at least 1 row to make it work
        .rolling(10, min_periods=1)
        .mean()
    )
)

# %%
subset_df.assign(
    rolling_births=lambda d: (
        d['births']
        # Do not use current row, only previous ones
        .shift(1)
        # Do rolling average of 10 rows, at least 1 row to make it work
        .rolling(10, min_periods=1)
        .mean()
    )
)
# %%
subset_df

# %% Same as before, but use 10D period instead of last 10 rows
(
    subset_df
    .set_index('date')
    .assign(rolling_births=lambda d: d['births'].shift(1).rolling('10D', min_periods=1).mean())
)

# %% This function may be used to do the same in a groupby manner


def calc_rolling_mean(dataf, column=None, setting='30D'):
    return (
        dataf
        .groupby('state')[column]
        .transform(lambda d: d.shift(1).rolling(setting, min_periods=1).mean())
    )


# %% Create a df with all data, no filter
unfilter_df = (
    df[['state', 'date', 'births']]
    .assign(date=lambda d: pd.to_datetime(d['date'], format="%Y-%m-%d"))
)

# %% This does the whole process of adding the rolling function
(
    unfilter_df
    .set_index('date')
    .assign(rolling_births=lambda d: calc_rolling_mean(d, column='births'))
    .reset_index()
    .sort_values(["state", "date"])
)

# %%
unfilter_df.births.mean()
# %%
unfilter_df.births.rolling(window=1).__getattr__('mean')()

# %%
(
    unfilter_df
    .set_index('date')
    .assign(rolling_births=lambda d: rolling_fn(
        d,
        column='births',
        groupby_cols='state',
        function='mean',
        setting='30D',
    ))
    .reset_index()
    .sort_values(["state", "date"])
    .query('state == "CA"')
)

# %%
(
    unfilter_df
    .set_index('date')
    .head(10000)
    .assign(rolling_births=lambda d: rolling_fn(
        d,
        column='births',
        groupby_cols='state',
        # function='mean',
        setting='30D',
        function=np.quantile,
        q=0.25
    ))
    .reset_index()
    .sort_values(["state", "date"])
)

# %%
ags = (
    unfilter_df
    .set_index('date')
    .groupby('state')['births']
    .shift(1)
    .rolling('30D', min_periods=1)
    .agg({'mean': 'mean', 'median': 'median', 'std': 'std'})
    .reset_index()
    # .transform(
    #     lambda d: (
    #         d
    #         .shift(1)
    #         .rolling('30D', min_periods=1)
    #         .agg(['mean', 'median'])
    #         # .agg({'mean': np.mean})
    #     )
    # )
)
# %%
pd.concat([unfilter_df, ags], axis=1).query("state == 'CA'")
# %%
