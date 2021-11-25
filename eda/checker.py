import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy, kruskal


def check_train_test(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.4):
    for col in train.columns:
        stat = ks_2samp(train[col].values, test[col].values).statistic
        if stat > threshold:
            print("Col:", col, " K-S test ", np.round(stat, decimals=2))
