import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        transformed = super().transform(X)
        n_new_cols = np.shape(transformed)[1] - len(self.columns)
        columns = list(self.columns.copy())
        for i in range(n_new_cols):
            columns.append(f"col_{i}_imputed")
        return pd.DataFrame(transformed, columns=columns)

class PandasStandardScaler(StandardScaler):
    """A wrapper around `StandardScaler` to return data frames with columns.
    """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)


# print(
#     PandasSimpleImputer(strategy="median", add_indicator=True).fit_transform(pd.DataFrame(dict(a=[1, None, 3], b=[None, 5, 6], c=[7, 8, 9], d=["a", "b", "c"])))
# )
