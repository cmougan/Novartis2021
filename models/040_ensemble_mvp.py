# %%
import pandas as pd
import sys
sys.path.append("../")

from tools.postprocessing import clip_first_month
from metrics.metric_participants import (ComputeMetrics, print_metrics)
from memo import memlist, grid, Runner


files = [
    # "empty_extractor_target_encoder",
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

def mix(d1, d2, weight, mix_interval=True, mix_sales=True, sales_winner=1, interval_winner=1):
    d = d1.copy()
    if mix_sales:
        d['sales'] = d1['sales'] * weight + d2['sales'] * (1 - weight)
    elif sales_winner == 1:
        d['sales'] = d1['sales']
    else:
        d['sales'] = d2['sales']
    
    if mix_interval:
        d['upper'] = d1['upper'] * weight + d2['upper'] * (1 - weight)
        d['lower'] = d1['lower'] * weight + d2['lower'] * (1 - weight)
    elif interval_winner == 1:
        d['upper'] = d1['upper']
        d['lower'] = d1['lower']
    else:
        d['upper'] = d2['upper']
        d['lower'] = d2['lower']

    return d
# %%

data = []


@memlist(data=data)
def mixing_output(weight, submission_1, submission_2, mix_interval=True, mix_sales=True, sales_winner=1, interval_winner=1):
    mixed = mix(
        submissions[submission_1]['val'], 
        submissions[submission_2]['val'],
        weight,
        mix_interval=mix_interval,
        mix_sales=mix_sales,
        sales_winner=sales_winner,
        interval_winner=interval_winner
    )

    metrics = ComputeMetrics(mixed, sales_train, ground_truth_val)
    return {"accuracy": metrics[0], "deviation": metrics[1]}
# %%

# data = []
settings = grid(weight=[0, 0.25, 0.5, 0.75, 1.0], submission_1=files, submission_2=files)

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
df_results.head(20)


# %%
