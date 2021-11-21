import pandas as pd



def rolling_fn(
    dataf: pd.DataFrame,
    groupby_cols: str = None,
    column: str = None,
    # function: str = 'mean',
    setting: str = '30D',
    shift_periods: int = 1,
) -> pd.Series:
    return (
        dataf
        .groupby(groupby_cols)[column]
        .transform(
            lambda d: (
                d
                .shift(shift_periods)
                .rolling(setting, min_periods=1)
                .mean()
            )
        )
    )