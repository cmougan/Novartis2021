# %%
import pandas as pd
import sys
sys.path.append("../")

from tools.postprocessing import clip_first_month, postprocess_submissions, crop_max
from metrics.metric_participants import (ComputeMetrics, print_metrics)
from memo import memlist, grid, Runner, random_grid


files = [
    # "empty_extractor_target_encoder",
    "gbm_time_evol",
    # "linear_model_time_evol",
    "linear_model_simple",
    "linear_model_grouped",
    "linear_model_feat_plus",
    "NN",
    "sklearn_gbm",
    "skgbm_time_evol"
    # "marc_magic_transforms_166",
    # "linear_winner_feat",
]
validations = {}
submissions = {}


df_full = pd.read_csv("../data/split.csv")
ground_truth_val = df_full.query("validation == 1").loc[:, ["month", "region", "brand", "sales"]]
sales_train = pd.read_csv("../data/data_raw/sales_train.csv")

# %%
def merge_two_submissions(df_accuracy, df_interval):
    df_interval = df_interval.copy().sort_values(['month', 'region', 'brand'])
    df_accuracy = df_accuracy.copy().sort_values(['month', 'region', 'brand'])
    df_interval['sales'] = df_accuracy['sales']
    return df_interval

# %%
submissions = {}

for file in files:
    submissions[file] = {}
    submissions[file]['val'] = clip_first_month(pd.read_csv(f"../data/validation/{file}_val.csv"))
    submissions[file]['submission'] = clip_first_month(pd.read_csv(f"../submissions/{file}.csv"))

# %%

submissions_nn_val = (
    submissions['NN']['val']
    .merge(
        submissions['linear_model_feat_plus']['val'],
        how='left',
        on=['month', 'region', 'brand']
    )
)

submissions_nn = (
    submissions['NN']['submission']
    .merge(
        submissions['linear_model_feat_plus']['submission'],
        how='left',
        on=['month', 'region', 'brand']
    )
)

def fix_nn_submission(submissions_nn):
    submissions_nn.sales_x = submissions_nn.sales_y
    part_1 = (
        submissions_nn
        .query('lower_x != 0 or upper_x != 0')
        .drop(columns=['sales_y', 'lower_y', 'upper_y'])
        .rename(columns={"sales_x": "sales", "upper_x": "upper", "lower_x": "lower"})
    )

    part_2 = (
        submissions_nn
        .query('lower_x == 0 and upper_x == 0')
        .drop(columns=['sales_x', 'lower_x', 'upper_x'])
        .rename(columns={"sales_y": "sales", "upper_y": "upper", "lower_y": "lower"})
    )

    return pd.concat([part_1, part_2]).sort_values(['month', 'region', 'brand']).reset_index(drop=True)

submissions['NN']['val'] = fix_nn_submission(submissions_nn_val)
submissions['NN']['submission'] = fix_nn_submission(submissions_nn)

# %%
def mix(d1, d2, weight_sales, weight_interval):
    d = d1.copy()
    d['sales'] = d1['sales'] * weight_sales + d2['sales'] * (1 - weight_sales)
    d['upper'] = d1['upper'] * weight_interval + d2['upper'] * (1 - weight_interval)
    d['lower'] = d1['lower'] * weight_interval + d2['lower'] * (1 - weight_interval)

    return d


def double_mix(d1, d2, d3, weight_sales_1, weight_interval_1, weight_sales_2, weight_interval_2):
    mix_1 = mix(d1, d2, weight_sales_1, weight_interval_1)
    return mix(mix_1, d3, weight_sales_2, weight_interval_2)


# %%

data_double = []


@memlist(data=data_double)
def mixing_output_double(submission_1, submission_2, submission_3, postprocess=True, weight_sales_1=0.5, weight_sales_2=0.5, weight_interval_1=0.5, weight_interval_2=0.5):
    # Don't want to do dummy ensembles
    if submission_1 == submission_2:
        return {"accuracy": 1000, "deviation": 1000}
    if submission_1 == submission_3:
        return {"accuracy": 1000, "deviation": 1000}
    if submission_2 == submission_3:
        return {"accuracy": 1000, "deviation": 1000}

    mixed = double_mix(
        submissions[submission_1]['val'], 
        submissions[submission_2]['val'],
        submissions[submission_3]['val'],
        weight_sales_1=weight_sales_1,
        weight_interval_1=weight_interval_1,
        weight_sales_2=weight_sales_2,
        weight_interval_2=weight_interval_2
    )

    if postprocess:
        mixed = postprocess_submissions(mixed)

    metrics = ComputeMetrics(mixed, sales_train, ground_truth_val)
    return {"accuracy": metrics[0], "deviation": metrics[1]}
# %%

# data = []
settings = random_grid(
    n=1000,
    weight_sales_1=[0, 0.25, 0.5, 0.75, 1.0],
    weight_interval_1=[0, 0.25, 0.5, 0.75, 1.0],
    weight_sales_2=[0, 0.25, 0.5, 0.75, 1.0],
    weight_interval_2=[0, 0.25, 0.5, 0.75, 1.0],
    submission_1=files,
    submission_2=files,
    submission_3=files,
    # postprocess=[True, False], 
    # mix_interval=[True, False], mix_sales=[True, False], sales_winner=[1, 2], interval_winner=[1, 2]
)

runner = Runner()
runner.run(
    func=mixing_output_double,
    settings=settings, 
)

# %%
df_results = pd.DataFrame.from_records(data_double).sort_values(by=['deviation'])

# %%
df_results.groupby(['accuracy', 'deviation'], as_index=False).first().sort_values(by=['deviation']).head(20)

# %%
df_results.groupby(['accuracy', 'deviation'], as_index=False).first().sort_values(by=['accuracy']).head(20)


# %%
accuracy_winner = df_results.groupby(['accuracy', 'deviation'], as_index=False).first().sort_values(by=['accuracy']).head(1)

accuracy_winner_dfs = {}
for key in ['val', 'submission']:
    accuracy_winner_dfs[key] = double_mix(
        d1=submissions[accuracy_winner.submission_1.values[0]][key],
        d2=submissions[accuracy_winner.submission_2.values[0]][key],
        d3=submissions[accuracy_winner.submission_3.values[0]][key],
        weight_sales_1=accuracy_winner.weight_sales_1.values[0],
        weight_interval_1=accuracy_winner.weight_interval_1.values[0],
        weight_sales_2=accuracy_winner.weight_sales_2.values[0],
        weight_interval_2=accuracy_winner.weight_interval_2.values[0],
    ).pipe(postprocess_submissions)


# %%
interval_winner = df_results.groupby(['accuracy', 'deviation'], as_index=False).first().sort_values(by=['deviation']).head(1)

interval_winner_dfs = {}
for key in ['val', 'submission']:
    interval_winner_dfs[key] = double_mix(
        d1=submissions[interval_winner.submission_1.values[0]][key],
        d2=submissions[interval_winner.submission_2.values[0]][key],
        d3=submissions[interval_winner.submission_3.values[0]][key],
        weight_sales_1=interval_winner.weight_sales_1.values[0],
        weight_interval_1=interval_winner.weight_interval_1.values[0],
        weight_sales_2=interval_winner.weight_sales_2.values[0],
        weight_interval_2=interval_winner.weight_interval_2.values[0],
    ).pipe(postprocess_submissions)
# .pipe(print_metrics, sales_train, ground_truth_val)
# %%

total_winner_dfs = {}

for key in ['val', 'submission']:
    total_winner_dfs[key] = merge_two_submissions(
        df_accuracy=accuracy_winner_dfs[key],
        df_interval=interval_winner_dfs[key],
    )
# total_winner = merge_two_submissions(df_accuracy=accuracy_winner, df_interval=interval_winner)

# %%
total_winner_dfs['val'].pipe(print_metrics, sales_train, ground_truth_val)


# %%
submission_name = "ensemble_3_model"
total_winner_dfs['val'].sort_values(['month', 'region', 'brand']).to_csv(f"../data/validation/{submission_name}_val.csv", index=False)
total_winner_dfs['submission'].sort_values(['month', 'region', 'brand']).to_csv(f"../submissions/{submission_name}.csv", index=False)


# %%
double_mix(
    d1=submissions['linear_model_grouped']['val'],
    d2=submissions['skgbm_time_evol']['val'],
    d3=submissions['linear_model_feat_plus']['val'],
    weight_sales_1=0.75,
    weight_interval_1=0.25,
    weight_sales_2=0.75,
    weight_interval_2=0.25,
).pipe(print_metrics, sales_train, ground_truth_val)
# %%
new_submissions = {}

for file in ['e_gbms_lm', 'e_time_gbms_lm']:
    new_submissions[file] = {}
    new_submissions[file]['val'] = clip_first_month(pd.read_csv(f"../data/validation/{file}_val.csv")).drop(columns=['max_sales'])
    new_submissions[file]['submission'] = clip_first_month(pd.read_csv(f"../submissions/{file}.csv")).drop(columns=['max_sales'])


# %%

(
    merge_two_submissions(
        df_accuracy=new_submissions['e_gbms_lm']['val'],
        df_interval=new_submissions['e_time_gbms_lm']['val'],
    )
    .assign(upper=lambda x: x['upper'] * 0.9)
    .pipe(print_metrics, sales_train, ground_truth_val)
)
# %%
best_merged = merge_two_submissions(
    df_accuracy=new_submissions['e_gbms_lm']['submission'],
    df_interval=new_submissions['e_time_gbms_lm']['submission'],
).assign(upper=lambda x: x['upper'] * 0.9)

# %%
submission_name = 'best_submission_02'
(
    best_merged
    .sort_values(['month', 'region', 'brand'])
    .to_csv(f"../submissions/{submission_name}.csv", index=False)
)

# %%
