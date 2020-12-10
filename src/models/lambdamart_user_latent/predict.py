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
parser = argparse.ArgumentParser(description='Use LambdaMART example model to predict on validation set.')
parser.add_argument('-r',
                    '--run_id',
                    type=str, 
                    help='Run ID for saving models')

parser.add_argument('-d',
                    '--dataset', 
                    nargs = '?',
                    type = str,
                    help='data_directory',
                    default = '/scratch/abh466/sad_data/raw/full/test'
                   )

parser.add_argument('-l',
                    '--latent', 
                    nargs = '?',
                    type = str,
                    help='embedding_directory',
                    default = '/scratch/abh466/sad_data/processed/full/test/user_latent_test.parquet'
                   )

parser.add_argument('-s',
                    '--split', 
                    nargs = '?',
                    type = str,
                    help='type of split',
                    default = 'test'
                   )
args = parser.parse_args()

features_path = 'src/data/schemas/output_data_schemas.json'

def __main__():
    mlflow.start_run(run_id=args.run_id)

    data = read_parquet(args.dataset)
    latents = pd.read_parquet(args.latent)

    print('finish read')
    data.dropna(subset=['user_id'],inplace=True)
    print('finish drop')
    # set display_rank to a constant to avoid feature leakage
    data['display_rank'] = 10
    with open(features_path, 'r') as features:
        model_feature_schemas = json.load(features)
        model_features = [f['name'] for f in model_feature_schemas if f['train']]

    latent_features = ['latent_{}'.format(i) for i in range(200)]
    for i in latent_features:
        model_features.append(i)
    
    data = data.join(latents, on = 'user_id', how = 'left', lsuffix = '_left', rsuffix = '_right')
    print('finish join')

    X, y, qid = feature_label_split(data, model_features, qid='search_request_id')
    X = X.astype('float')

    model= mlflow.xgboost.load_model('runs:/{}/model.json'.format(args.run_id))

    pred_array = data[['search_request_id', 'hotel_id']].copy()

    pred_array['score'] = model.predict(xgb.DMatrix(X))
    pred_array['rank'] = pred_array\
        .groupby('search_request_id')\
        ['score']\
        .rank(ascending=False)

    predictions_path = 'predictions/{}/full/{}'.format(args.run_id, args.split)
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
