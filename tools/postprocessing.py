import pandas as pd

def postprocess_predictions(predictions):
    predictions = predictions.copy()
    for p, pred in predictions.items():
        pred[pred < 0] = 0
        predictions[p] = pred
    l = predictions[0.1]
    m = predictions[0.5]
    u = predictions[0.9]
    idpredictions = l > m
    l[idpredictions] = m[idpredictions] - 0.001
    idpredictions = u < m
    u[idpredictions] = m[idpredictions] + 0.001
    predictions[0.1] = l
    predictions[0.5] = m
    predictions[0.9] = u
    return predictions


def postprocess_submissions(df):

    df = df.copy()
    for c in ['sales', 'lower', 'upper']:
        df.loc[df[c] < 0, c] = 0
        
    bad_lower_index = df['sales'] < df['lower']
    df.loc[bad_lower_index, 'lower'] = df.loc[bad_lower_index, 'sales'] - 0.01

    bad_upper_index = df['sales'] > df['upper']
    df.loc[bad_upper_index, 'upper'] = df.loc[bad_upper_index, 'sales'] + 0.01

    return df


def clip_first_month(df, month='2020-07', brand='brand_1'):
    df = df.copy()
    idx = (df['month'] == month) & (df['brand'] == brand)
    df.loc[idx, 'sales'] = 0
    df.loc[idx, 'lower'] = 0
    df.loc[idx, 'upper'] = 0
    return df

def crop_max(df_valid):
    
    df_valid = df_valid.copy()

    maxes = (
        pd.read_csv("../data/data_raw/sales_train.csv")
        .query("brand.isin(['brand_1', 'brand_2'])", engine='python')
        .groupby(["month", "brand"], as_index=False)
        .agg({"sales": "max"})
        .rename(columns={"sales": "max_sales"})
    )

    df_valid = df_valid.merge(maxes, on=["month", "brand"])
    df_valid["sales"] = df_valid["sales"].clip(lower=0, upper=df_valid["max_sales"])
    df_valid["upper"] = df_valid["upper"].clip(lower=0, upper=df_valid["max_sales"])

    df_valid.drop(columns=["max_sales"])

    return df_valid