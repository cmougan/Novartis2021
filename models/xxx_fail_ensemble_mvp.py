# %%
import pandas as pd
import sys
sys.path.append("../")

from tools.postprocessing import clip_first_month
from metrics.metric_participants import (ComputeMetrics, print_metrics)


files = [
    "empty_extractor_target_encoder",
    "gbm_time_evol",
    "linear_model_time_evol",
    "linear_model_simple",
    "linear_model_grouped",
]
validations = {}
submissions = {}


df_full = pd.read_csv("../data/split.csv")
ground_truth_val = df_full.query("validation == 1").loc[:, ["month", "region", "brand", "sales"]]
sales_train = pd.read_csv("../data/data_raw/sales_train.csv")

class Submission(object):
    def __init__(self, name):
        self.name = name
        self.val_df = pd.read_csv(f"../data/validation/{self.name}_val.csv")
        self.val_df = clip_first_month(self.val_df)
        self.submission_df = pd.read_csv(f"../submissions/{self.name}.csv")
        self.submission_df = clip_first_month(self.submission_df)
    
    def plus(self, other):
        output = Submission(self.name)
        output.val_df['sales'] += other.val_df['sales']
        output.submission_df['sales'] += other.submission_df['sales']

        output.val_df['lower'] += other.val_df['lower']
        output.submission_df['lower'] += other.submission_df['lower']

        output.val_df['upper'] += other.val_df['upper']
        output.submission_df['upper'] += other.submission_df['upper']

        return output

    def dot(self, other: float):
        output = Submission(self.name)
        print(other)
        print(output.val_df['sales'].head())
        print((output.val_df['sales'] * other).head())
        output.val_df['sales'] = output.val_df['sales'] * other
        print("After")
        print(output.val_df['sales'].head())
        # output.submission_df['sales'] *= other

        output.val_df['lower'] = output.val_df['lower'] * other
        # output.submission_df['lower'] *= other

        output.val_df['upper']*= output.val_df['upper'] * other
        # output.submission_df['upper'] *= other

        return output
    
    def eval_preds(self):
        return ComputeMetrics(self.val_df, sales_train, ground_truth_val)
# %%
submissions = {}

for file in files:
    submissions[file] = Submission(file)

# %%
(submissions["linear_model_simple"] + submissions["linear_model_simple"]).dot(0.5).val_df

# %%
submissions["linear_model_simple"].val_df
# %%