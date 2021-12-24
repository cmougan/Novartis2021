# %%
import pandas as pd
from copy import deepcopy
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
    def __init__(
        self,
        name,
        keys=('sales', 'lower', 'upper'),
        alphas=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ):
        self.name = name
        self.val_df = pd.read_csv(f"../data/validation/{self.name}_val.csv")
        self.val_df = clip_first_month(self.val_df)
        self.submission_df = pd.read_csv(f"../submissions/{self.name}.csv")
        self.submission_df = clip_first_month(self.submission_df)
        self.keys = keys
        self.alphas = alphas
    
    def __repr__(self):
        return self.val_df.head(5).to_string()

    def __add__(self, other):
        output = deepcopy(self)
        for key in self.keys:
            output.submission_df[key] += other.submission_df[key]
            output.val_df[key] += other.val_df[key]

        return output

    def __mul__(self, other: float):
        output = deepcopy(self)
        for key in self.keys:
            output.submission_df[key] *= other
            output.val_df[key] *= other

        return output
    
    def optimize(self, other):
        min_alpha = 0
        best_submission = self
        for alpha in self.alphas:
            submission = (self * (1 - alpha)) + (other * alpha)
            if submission <= best_submission:
                min_alpha = alpha
                best_submission = submission
        return best_submission, min_alpha
        
    def __le__(self, other):
        return self.eval_preds()[1] <= other.eval_preds()[1]

    def eval_preds(self):
        return ComputeMetrics(self.val_df, sales_train, ground_truth_val)
# %%
submissions = {}

for file in files:
    submissions[file] = Submission(file)

# %%
(
    (
        submissions["gbm_time_evol"] + submissions["linear_model_simple"]
    ) * 0.5
).eval_preds()


# %%
submissions["linear_model_simple"].eval_preds()
# %%
submissions["linear_model_simple"].optimize(submissions["gbm_time_evol"])
# %%
submissions["linear_model_simple"]
