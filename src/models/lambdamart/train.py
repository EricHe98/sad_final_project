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

data_path = 'data/raw/'
features_path = 'src/data/schemas/output_data_schemas.json'

model_directory = 'models/example_model_lambdamart'

def __main__():
    data = read_parquet('data/raw/', num_partitions=5)

    with open(features_path, 'r') as features:
        model_feature_schemas = json.load(features)
        model_features = [f['name'] for f in model_feature_schemas if f['train']]

    X, y, qid = feature_label_split(data, model_features, qid='search_request_id')
    X = X.astype('float')

    with mlflow.start_run(run_name='example_small_lambdamart'):
        run_id = mlflow.active_run().info.run_id
        print('MLFlow Run ID is: {}'.format(run_id))
        model = XGBRanker(objective='rank:ndcg', verbosity=2, n_jobs=-1, n_estimators=10)
        mlflow.log_params(model.get_params())
        time_start = dt.datetime.now()
        model.fit(X, y , qid)
        time_end = dt.datetime.now()

        train_time = (time_end - time_start).total_seconds()
        mlflow.log_metric('training_time', train_time)
        print('Model trained in {}'.format(train_time))

        mlflow.xgboost.log_model(model, 'example_small_lambdamart.json')
    
if __name__ == '__main__':
    __main__()