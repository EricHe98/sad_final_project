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
parser.add_argument('run', type=str, 
    help='Run ID for saving models')

args = parser.parse_args()
run_id = args.run

data_path = 'data/raw/'
features_path = 'src/data/schemas/output_data_schemas.json'

def __main__():
	mlflow.start_run(run_id=run_id)

	data = read_parquet('data/val/')
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

if __name__ == '__main__':
	__main__()
