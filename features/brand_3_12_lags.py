import pandas as pd

data = pd.read_csv("../data/features/brand_3_12_market_features.csv")


def grouped(data, col: str = "", shifter: int = 1):
    aux = data.groupby(["month", "region"])[col].sum().shift(1).reset_index()
    title = col + "shift" + str(shifter)
    aux.columns = ["month", "region", title]

    return pd.merge(data, aux, on=["month", "region"])


for col in ["sales_brand_3", "sales_brand_3_market", "sales_brand_12_market"]:
    for i in range(1, 12):
        data = grouped(data, col=col, shifter=i)

reg3 = data.groupby("region")["sales_brand_3_market"].sum().reset_index()
reg12 = data.groupby("region")["sales_brand_12_market"].sum().reset_index()

reg3.columns = ["region", "sales_brand_3_market_per_region"]

reg12.columns = ["region", "sales_brand_12_market_per_region"]

data = pd.merge(data, reg3, on="region")
data = pd.merge(data, reg12, on="region")

data.to_csv("../data/features/brand_3_12_market_features_lagged.csv", index=False)
