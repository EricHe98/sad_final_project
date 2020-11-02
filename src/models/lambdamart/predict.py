import os
import json
import pickle
import sys
import traceback
import datetime as dt

import numpy as np 
import pandas as pd

import xgboost as xgb
from xgboost import XGBRanker, XGBClassifier

from sklearn.linear_model import LogisticRegression
import mlflow

import argparse
parser = argparse.ArgumentParser(description='Use LambdaMART example model to predict on validation set.')
parser.add_argument('run', type=str, 
    help='Run ID for saving models')

args = parser.parse_args()
run_id = args.run

data_path = 'data/raw/'
features_path = 'src/data/schemas/output_data_schemas.json'

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
	mlflow.start_run(run_id=run_id)

	data = read_parquet('data/raw/', num_partitions=5)
	with open(features_path, 'r') as features:
	    model_feature_schemas = json.load(features)
	    model_features = [f['name'] for f in model_feature_schemas if f['train']]

	X, y, qid = feature_label_split(data, model_features, qid='search_request_id')
	X = X.astype('float')

	model= mlflow.xgboost.load_model('runs:/{}/example_small_lambdamart.json'.format(run_id))

	pred_array = data[['search_request_id', 'hotel_id']].copy()

	pred_array['score'] = model.predict(xgb.DMatrix(X))
	pred_array['rank'] = pred_array\
	    .groupby('search_request_id')\
	    ['score']\
	    .rank(ascending=False)

	predictions_path = 'predictions/{}'.format(run_id)
	predictions_file = os.path.join(predictions_path, 'predictions.parquet')

	if not os.path.exists(predictions_path):
	    os.mkdir(predictions_path)
	    
	pred_array.to_parquet(predictions_file)

	mlflow.log_artifact(predictions_file)

if __name__ == '__main__':
	__main__()
