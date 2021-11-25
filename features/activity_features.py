# %% Imports
import pandas as pd
import re

def flatten_columns(df):
    df.columns = ["_".join(df) for df in df.columns.ravel()]
    df.columns = [re.sub(r'_$', '', col) for col in df.columns]
    return df

def rename_cols_3m(df):
    df.columns = [f"{col}_3m" if col not in INDEX else col for col in df.columns]
    return df

def count_tier_features(df, column=None):
    if column:
        return ( 
            df
            .assign(inverse_tier=lambda x: 3 - x.tier)
            .groupby(INDEX + [column], as_index=False)
            .agg({
                'count': 'sum', 
                'tier': lambda x: x.isnull().sum(),
                'inverse_tier': 'sum'
            })
            .rename(columns={'tier': 'null_tiers'})
            .pivot(index=INDEX, columns=[column], values=['count', 'null_tiers', 'inverse_tier'])
            .reset_index()
            .pipe(flatten_columns)
        )  
    else:
        return ( 
            df
            .assign(inverse_tier=lambda x: 3 - x.tier)
            .groupby(INDEX, as_index=False)
            .agg({
                'count': 'sum', 
                'tier': lambda x: x.isnull().sum(),
                'inverse_tier': 'sum'
            })
            .rename(columns={'tier': 'null_tiers'})
        )

df_full = pd.read_csv('../data/split.csv')
df_activity = pd.read_csv('../data/data_raw/activity.csv')
hcps = pd.read_csv('../data/data_raw/hcps.csv')

INDEX = ['month', 'region', 'brand']
# %%
df_activity_hcp = (
    df_activity
    .merge(hcps.drop(columns=['region', 'specialty']), how='left', on='hcp')
    .rename(columns={'month': 'activity_month'})
    .assign(activity_month_date=lambda x: x.activity_month.apply(lambda x: pd.to_datetime(x, format='%Y-%m')))
)

# %%
df_leaked_activity = (
    df_full
    .assign(month_date=lambda x: x.month.apply(lambda x: pd.to_datetime(x, format='%Y-%m')))
    .merge(df_activity_hcp, how='left', on=['region', 'brand'])
)

df_full_activity = (
    df_leaked_activity
    # TODO: Use leakage in here
    .query('month >= activity_month')
)

df_lag_3m = df_full_activity[
    df_full_activity.activity_month_date + pd.DateOffset(months=3) >= df_full_activity.month_date
]

# %%
basic_features = count_tier_features(df_full_activity)
basic_features_channel = count_tier_features(df_full_activity, 'channel')
basic_features_specialty = count_tier_features(df_full_activity, 'specialty')

# %%
basic_features_3m = rename_cols_3m(count_tier_features(df_lag_3m))
basic_features_channel_3m = rename_cols_3m(count_tier_features(df_lag_3m, 'channel'))
basic_features_specialty_3m = rename_cols_3m(count_tier_features(df_lag_3m, 'specialty'))

# %%
df_activity_features = (
    df_full
    .merge(basic_features, on=INDEX, how='left')
    .merge(basic_features_channel, on=INDEX, how='left')
    .merge(basic_features_specialty, on=INDEX, how='left')
    .merge(basic_features_3m, on=INDEX, how='left')
    .merge(basic_features_channel_3m, on=INDEX, how='left')
    .merge(basic_features_specialty_3m, on=INDEX, how='left')
)
# %% Save to features/activity_features.csv
df_activity_features.to_csv('../data/features/activity_features.csv', index=False)
