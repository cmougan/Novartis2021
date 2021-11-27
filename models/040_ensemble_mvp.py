# %%
import pandas as pd
import sys
sys.path.append("../")

from tools.postprocessing import clip_first_month
from metrics.metric_participants import (ComputeMetrics, print_metrics)
from memo import memlist, grid, Runner


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

# %%
submissions = {}

for file in files:
    submissions[file] = {}
    submissions[file]['val'] = clip_first_month(pd.read_csv(f"../data/validation/{file}_val.csv"))
    submissions[file]['submission'] = clip_first_month(pd.read_csv(f"../submissions/{file}.csv"))

# %%

def mix(d1, d2, weight):
    d = d1.copy()
    d['sales'] = d1['sales'] * weight + d2['sales'] * (1 - weight)
    d['upper'] = d1['upper'] * weight + d2['upper'] * (1 - weight)
    d['lower'] = d1['lower'] * weight + d2['lower'] * (1 - weight)
    return d
# %%

data = []


@memlist(data=data)
def mixing_output(weight):
    mixed = mix(
        submissions['linear_model_simple']['val'], 
        submissions['gbm_time_evol']['val'],
        weight
    )

    metrics = ComputeMetrics(mixed, sales_train, ground_truth_val)
    return {"accuracy": metrics[0], "deviation": metrics[1]}
# %%
settings = grid(weight=[0, 0.1, 0.2, 0.5, 0.8, 1.0])

runner = Runner()
runner.run(
    func=mixing_output,
    settings=settings, 
)

# %%
df_results = pd.DataFrame.from_records(data).sort_values(by=['deviation'])
optimal_weight = float(df_results.head(1).weight)
# %%
optimal_weight

# %%
df_results

