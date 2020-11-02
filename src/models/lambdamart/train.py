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
from xgboost import XGBRanker, XGBClassifier

from sklearn.linear_model import LogisticRegression
import mlflow

data_path = 'data/raw/'
features_path = 'src/data/schemas/output_data_schemas.json'

model_directory = 'models/example_model_lambdamart'

def read_parquet(data_path, num_partitions=None, random=True, verbose=True):
    files = os.listdir(data_path)
    if random:
        import random
        random.shuffle(files)
    if num_partitions is None:
        num_partitions = len(files)
        
    data = []
    num_reads = 0
    for file_path in files:
        if num_reads >= num_partitions:
            break
        root, ext = os.path.splitext(file_path)
        # exclude non-parquet files (e.g. gitkeep, other folders)
        if ext == '.parquet':
            fp = os.path.join(data_path, file_path)
            if verbose:
                print('Reading in data from {}'.format(fp))
            data.append(pd.read_parquet(os.path.join(data_path, file_path)))
            if verbose:
                print('Data of shape {}'.format(data[-1].shape))
            num_reads += 1
        else: 
            continue
    data = pd.concat(data, axis=0)
    if verbose:
        print('Total dataframe of shape {}'.format(data.shape))
    return data

def feature_label_split(data, model_features, label='label', qid='qid'):
    # assumes data of same QIDs are grouped together
    X = data[model_features]
    y = data[label]
    qid = data[qid].value_counts(sort=False).sort_index()
    return X, y, qid

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