import os
import json
import pickle
import sys
import traceback
import datetime as dt
import click

import numpy as np 
import pandas as pd

import xgboost as xgb
from xgboost import XGBRanker

from src.modules.modules_pandas import read_parquet, feature_label_split

import mlflow
import argparse 

parser = argparse.ArgumentParser(description='Use LambdaMART example model to predict on validation set.')
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full', 'downsampled'],
    help='which dataset to train the model (small_100, small_all, all) to be included in MLFlow name')
parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
    help='which split of the dataset to train on (train, val, test)')

args = parser.parse_args()
data_path = 'data/raw'

features_path = 'src/data/schemas/output_data_schemas.json'

def __main__():
    data = read_parquet(os.path.join(data_path, args.dataset, args.split))

    with open(features_path, 'r') as features:
        model_feature_schemas = json.load(features)
        model_features = [f['name'] for f in model_feature_schemas if f['train']]

    X, y, qid = feature_label_split(data, model_features, qid='search_request_id')
    X = X.astype('float')

    with mlflow.start_run(run_name='lambdamart'):
        run_id = mlflow.active_run().info.run_id
        print('MLFlow Run ID is: {}'.format(run_id))
        mlflow.log_param('dataset', args.dataset)
        mlflow.log_param('train_split', args.split)
        mlflow.log_param('model_name', 'lambdamart')
        mlflow.log_param('run_id', run_id)
        # n_jobs=-1 is supposed to mean use all cores available
        # however n_jobs=-1 seems to only be parallelized on latest xgboost version 1.3.0rc1
        # make sure to check, otherwise manually set n_jobs to number of cores available
        model = XGBRanker(objective='rank:ndcg', verbosity=2, n_jobs=-1, n_estimators=100)
        mlflow.log_params(model.get_params())
        time_start = dt.datetime.now()
        model.fit(X, y, qid)
        time_end = dt.datetime.now()

        train_time = (time_end - time_start).total_seconds()
        mlflow.log_metric('training_time', train_time)
        print('Model trained in {}'.format(train_time))

        mlflow.xgboost.log_model(model, 'model.json')
    
if __name__ == '__main__':
    __main__()