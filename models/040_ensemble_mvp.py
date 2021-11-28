# %%
import pandas as pd
import sys
sys.path.append("../")

from tools.postprocessing import clip_first_month, postprocess_submissions, crop_max
from metrics.metric_participants import (ComputeMetrics, print_metrics)
from memo import memlist, grid, Runner


files = [
    # "empty_extractor_target_encoder",
    "gbm_time_evol",
    "linear_model_time_evol",
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
def mixing_output(weight, submission_1, submission_2, postprocess=False, mix_interval=True, mix_sales=True, sales_winner=1, interval_winner=1):
    mixed = mix(
        submissions[submission_1]['val'], 
        submissions[submission_2]['val'],
        weight,
        mix_interval=mix_interval,
        mix_sales=mix_sales,
        sales_winner=sales_winner,
        interval_winner=interval_winner
    )

    if postprocess:
        mixed = postprocess_submissions(mixed)

    metrics = ComputeMetrics(mixed, sales_train, ground_truth_val)
    return {"accuracy": metrics[0], "deviation": metrics[1]}
# %%

# data = []
settings = grid(
    weight=[0, 0.25, 0.5, 0.75, 1.0],
    submission_1=files,
    submission_2=files,
    # postprocess=[True, False], 
    # mix_interval=[True, False], mix_sales=[True, False], sales_winner=[1, 2], interval_winner=[1, 2]
)

runner = Runner()
runner.run(
    func=mixing_output,
    settings=settings, 
)

# %%
df_results = pd.DataFrame.from_records(data).sort_values(by=['deviation'])

# %%
df_results.groupby(['accuracy', 'deviation'], as_index=False).first().sort_values(by=['deviation']).head(20)

# %%
df_results.groupby(['accuracy', 'deviation'], as_index=False).first().sort_values(by=['accuracy']).head(20)


# %%
ensemble_submission = mix(
    submissions['gbm_time_evol']['submission'],
    submissions['linear_model_feat_plus']['submission'],
    weight=0.5
).pipe(postprocess_submissions).pipe(crop_max)
# %%
ensemble_submission_val = mix(
    submissions['gbm_time_evol']['val'],
    submissions['linear_model_feat_plus']['val'],
    weight=0.5
).pipe(postprocess_submissions).pipe(crop_max)

# %%
submission_name = "e_gbm_evol_lm_feat_pls"
ensemble_submission_val.to_csv(f"../data/validation/{submission_name}_val.csv", index=False)
ensemble_submission.to_csv(f"../submissions/{submission_name}.csv", index=False)



# %%
dfs = {}
for key in ['val', 'submission']:

    mix_lms = mix(
        submissions['linear_model_feat_plus'][key],
        submissions['linear_model_time_evol'][key],
        weight=0.5,
        mix_interval=False,
        mix_sales=False,
        sales_winner=2,
        interval_winner=1
    )
    mix_gbm = mix(
        submissions['sklearn_gbm'][key],
        submissions['gbm_time_evol'][key],
        weight=0.5
    )
    full_mix = mix(
        mix_gbm,
        mix_lms,
        # submissions['linear_model_feat_plus'][key],
        weight=0.5
    ).pipe(postprocess_submissions).pipe(crop_max)

    dfs[key] = full_mix
    if key == 'val':
        full_mix.pipe(print_metrics, sales_train, ground_truth_val)


# %%
submission_name = "e_gbms_lm"
dfs['val'].sort_values(['month', 'region', 'brand']).to_csv(f"../data/validation/{submission_name}_val.csv", index=False)
dfs['submission'].sort_values(['month', 'region', 'brand']).to_csv(f"../submissions/{submission_name}.csv", index=False)

# %%
dfs = {}
for key in ['val', 'submission']:

    mix_lms = mix(
        submissions['linear_model_feat_plus'][key],
        submissions['linear_model_time_evol'][key],
        weight=0.5,
        mix_interval=False,
        mix_sales=False,
        sales_winner=2,
        interval_winner=1
    )
    mix_gbm_raw = mix(
        submissions['sklearn_gbm'][key],
        submissions['gbm_time_evol'][key],
        weight=0.5
    )

    mix_gbm = mix(
        submissions['skgbm_time_evol'][key],
        mix_gbm_raw,
        weight=0.5
    )
    full_mix = mix(
        mix_gbm,
        mix_lms,
        # submissions['linear_model_feat_plus'][key],
        weight=0.5
    ).pipe(postprocess_submissions).pipe(crop_max)

    dfs[key] = full_mix
    if key == 'val':
        full_mix.pipe(print_metrics, sales_train, ground_truth_val)


# %%
submission_name = "e_time_gbms_lm"
dfs['val'].sort_values(['month', 'region', 'brand']).to_csv(f"../data/validation/{submission_name}_val.csv", index=False)
dfs['submission'].sort_values(['month', 'region', 'brand']).to_csv(f"../submissions/{submission_name}.csv", index=False)

# %%
