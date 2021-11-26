# %% Imports
import pandas as pd
import re

def flatten_columns(df):
    df.columns = ["_".join(df) for df in df.columns.ravel()]
    df.columns = [re.sub(r'_$', '', col) for col in df.columns]
    return df

df_full = pd.read_csv('../data/split.csv')
df_activity = pd.read_csv('../data/data_raw/activity.csv')
df_rte_raw = pd.read_csv('../data/data_raw/rtes.csv')
hcps = pd.read_csv('../data/data_raw/hcps.csv')

INDEX = ['month', 'region', 'brand']


# %%
df_rte = (
    df_rte_raw
    .merge(hcps.loc[:, ["hcp", "tier"]], how='left', on='hcp')
    .assign(inverse_tier_1=lambda x: 4 - x.tier)
)

# %%
df_rte['date_last_opened'] = pd.to_datetime(df_rte.time_last_opened).dt.date
df_rte['date_sent'] = pd.to_datetime(df_rte.time_sent).dt.date

# %%
df_rte['month_last_opened'] = pd.to_datetime(df_rte.date_last_opened).dt.to_period('M')
df_rte['month_sent'] = pd.to_datetime(df_rte.date_sent).dt.to_period('M')

# %%

def aggs_rte_fn(df, columns=[]):
    idx = ['month_sent', 'region', 'brand']
    base_df = (
        df
        .assign(tier_openings=lambda x: x.inverse_tier_1 * x['no. openings'])
        .groupby(idx + columns, as_index=False)
        .agg({
            'tier_openings': 'sum',
            'no. openings': 'sum',
            'hcp': pd.Series.nunique,
        })
        .rename(columns={'hcp': 'hcp_distinct'})
    )

    if not columns:
        return base_df

    return (
        base_df
        .pivot(index=idx, columns=columns, values=['tier_openings', 'no. openings', 'hcp_distinct'])
        .reset_index()
        .pipe(flatten_columns)
    )


# %%
aggs_rte = aggs_rte_fn(df_rte)
aggs_rte_specialty = aggs_rte_fn(df_rte, ['specialty'])
aggs_rte_email_type = aggs_rte_fn(df_rte, ['email_type'])

# %%
def join_and_sum(df, aggs_df):
    return (
        df
        .drop(columns=['sales', 'validation'])
        .merge(aggs_df, how='left', on=['region', 'brand'])
        .query("month_sent <= month")
        .groupby(INDEX, as_index=False)
        .sum()
    )    

# %%
df_full_ = join_and_sum(df_full, aggs_rte)
df_full_specialty = join_and_sum(df_full, aggs_rte_specialty)
df_full_email_type = join_and_sum(df_full, aggs_rte_email_type)

# %%
(
    df_full
    .merge(df_full_, on=INDEX, how='left')
    .merge(df_full_specialty, on=INDEX, how='left')
    .merge(df_full_email_type, on=INDEX, how='left')
    .to_csv('../data/features/rte_basic_features.csv', index=False)
)
# %%
