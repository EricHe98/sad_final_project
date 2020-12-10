import os
import json
import pickle
import sys
import traceback
import datetime as dt
import click

import numpy as np 
import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.modules.modules_pandas import read_parquet, feature_label_split

import mlflow
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to train the model (small_100, small_all, all) to be included in MLFlow name')
parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
    help='which split of the dataset to train on (train, val, test)')

args = parser.parse_args()
data_path = 'data/raw'

features_path = 'src/data/schemas/output_data_schemas.json'
with open(features_path, 'r') as features:
    model_feature_schemas = json.load(features)
    model_features = [f['name'] for f in model_feature_schemas if f['train']]

model_features = ['hotel_cumulative_share', 'srq_price_zscore', 'previous_user_hotel_interaction',
    'srq_rewards_zscore', 'travel_intent', 'srq_distance_zscore', 'user_preferred_price']

id_features = ['search_request_id', 'hotel_id','user_id', 'label']

def __main__():
    data = read_parquet(os.path.join(data_path, args.dataset, args.split), columns= model_features + id_features)

    X, y, qid = feature_label_split(data, model_features, qid='search_request_id')
    X = X.astype('float').fillna(0)
    y = np.where(y >= 1, 1, y)

    with mlflow.start_run(run_name='logistic_regression'):
        run_id = mlflow.active_run().info.run_id
        print('MLFlow Run ID is: {}'.format(run_id))
        mlflow.log_param('dataset', args.dataset)
        mlflow.log_param('train_split', args.split)
        mlflow.log_param('model_name', 'logistic_regression')
        mlflow.log_param('run_id', run_id)
        # n_jobs=-1 is supposed to mean use all cores available
        # however n_jobs=-1 seems to only be parallelized on latest xgboost version 1.3.0rc1
        # make sure to check, otherwise manually set n_jobs to number of cores available
        model = LogisticRegression()
        mlflow.log_params(model.get_params())
        time_start = dt.datetime.now()
        model.fit(X, y)
        time_end = dt.datetime.now()

        train_time = (time_end - time_start).total_seconds()
        mlflow.log_metric('training_time', train_time)
        print('Model trained in {}'.format(train_time))

        mlflow.sklearn.log_model(model, 'model.json')
    
if __name__ == '__main__':
    __main__()
