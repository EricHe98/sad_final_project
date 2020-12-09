import os
import json
import pickle
import sys
import traceback
import datetime as dt

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBRanker

from src.modules.modules_pandas import read_parquet, feature_label_split

import mlflow

import argparse
parser = argparse.ArgumentParser(description='Use LambdaMART with embedings example model to predict on validation set.')
parser.add_argument('run_id', type=str,
    help='Run ID for saving models')
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to predict on (small_100, small_all, all)')
parser.add_argument('split', choices=['train', 'val', 'test'],
    help='which split of the dataset to predict on (train, val, test)')
parser.add_argument('--data', type=str, default='data/raw', required=False,
    help='path to data dir')
parser.add_argument('--emb', type=str, required=True,
    help='path to embeddings file')

args = parser.parse_args()

features_path = 'src/data/schemas/emb_output_data_schemas.json'

def transform_embeddings(emb_df):
    exp_emb_df = pd.DataFrame(np.column_stack(emb_df.values.T.tolist()))
    emb_col_names = ["hotel_id"] + [f"emb_{i}" for i in range(len(exp_emb_df.columns) - 1)]
    exp_emb_df.columns = emb_col_names

    return exp_emb_df, emb_col_names

def join_emb_data(emb_df, data, emb_col_names):
    data = pd.merge(data, emb_df, how='left', on='hotel_id')
    for n in emb_col_names:
        data[n].fillna(data[n].mean())
    return data

def __main__():
    mlflow.start_run(run_id=args.run_id)

    data = read_parquet(os.path.join(args.data, args.dataset, args.split))
    emb_df = pd.read_parquet(args.emb)
    emb_df, emb_col_names = transform_embeddings(emb_df)
    data = join_emb_data(emb_df, data, emb_col_names)

    # set display_rank to a constant to avoid feature leakage
    data['display_rank'] = 10
    with open(features_path, 'r') as features:
        model_feature_schemas = json.load(features)
        model_features = [f['name'] for f in model_feature_schemas if f['train']]

    X, y, qid = feature_label_split(data, model_features, qid='search_request_id')
    X = X.astype('float')

    model= mlflow.xgboost.load_model('runs:/{}/model.json'.format(args.run_id))

    pred_array = data[['search_request_id', 'hotel_id']].copy()

    pred_array['score'] = model.predict(xgb.DMatrix(X))
    pred_array['rank'] = pred_array\
        .groupby('search_request_id')\
        ['score']\
        .rank(ascending=False)

    predictions_path = 'predictions/{}/{}/{}'.format(args.run_id, args.dataset, args.split)
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    predictions_file = os.path.join(predictions_path, 'predictions.parquet')

    if not os.path.exists(predictions_path):
        os.mkdir(predictions_path)

    pred_array.to_parquet(predictions_file)

    # log 95th percentile latencies predicting on 2, 10, 50, 100, 500 results
    # It's pretty hard to get a consistent timing. There are a lot of confounds.
    # My convention here is to assume you have a Pandas DataFrame of the data ready to go
    N_RESULTS = [2, 10, 50, 100, 500]
    N_TRIALS = 5000
    for r in N_RESULTS:
        times = []
        for t in range(N_TRIALS):
            sample = X.sample(r)
            time_start = dt.datetime.now()
            sample_dmat = xgb.DMatrix(sample)
            model.predict(sample_dmat)
            time_end = dt.datetime.now()
            times.append((time_end - time_start).total_seconds() * 1000)
        pctile_95 = np.percentile(times, 95)
        print('95th percentile latency at {} results: {} ms'.format(r, pctile_95))
        mlflow.log_metric('latency_{}_results'.format(r), pctile_95)

if __name__ == '__main__':
    __main__()
