"""
Evaluate predictions given by a model.
Takes in paths to validation data and predictions data
	(assumes the data there is inner-joinable)
Predictions data must contain `search_request_id`, `hotel_id`, `score`, `rank` columns
Must supply `run_id` to join metrics back to MLFlow
"""
import os
import json
import sys
import datetime as dt

import numpy as np 
import pandas as pd

import mlflow

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')

from src.modules.modules_pandas import read_parquet, feature_label_split
import src.modules.letor_metrics as lm

import argparse
parser = argparse.ArgumentParser(description='Use LambdaMART example model to predict on validation set.')
parser.add_argument('run_id', type=str, 
    help='Run ID for saving models')
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to predict on (small_100, small_all, all)')
parser.add_argument('split', choices=['train', 'val', 'test'],
	help='which split of the dataset to predict on (train, val, test)')

args = parser.parse_args()
data_path = 'data/raw'

def __main__():
	data = read_parquet(os.path.join(data_path, args.dataset, args.split))

	mlflow.start_run(args.run_id)

	pred_file = 'predictions/{}/{}/{}/predictions.parquet'\
	    .format(args.run_id, args.dataset, args.split)
	pred = pd.read_parquet(pred_file)

	join_keys=['search_request_id', 'hotel_id']
	joined = data.set_index(join_keys)\
	    .join(pred.set_index(join_keys))\
	    .reset_index()

	ndcg_default = lm.ndcg(
	    joined,
	    groupby='search_request_id',
	    ranker='display_rank',
	    reverse=True)
	mlflow.log_metric('ndcg_default', ndcg_default.mean())

	ndcg_popularity = lm.ndcg(
	    joined,
	    groupby='search_request_id',
	    ranker='hotel_cumulative_share')
	mlflow.log_metric('ndcg_popularity', ndcg_popularity.mean())

	ndcg_model = lm.ndcg(
	    joined, 
	    groupby='search_request_id',
	    ranker='score')
	mlflow.log_metric('ndcg', ndcg_model.mean())

	tau_default = lm.tau(
	    joined,
	    groupby='search_request_id',
	    a='score',
	    b='display_rank')
	mlflow.log_metric('tau_default', tau_default.mean())

	tau_price = lm.tau(
	    joined,
	    groupby='search_request_id',
	    a='score',
	    b='price_rank')
	mlflow.log_metric('tau_price', tau_price.mean())

	mlflow.end_run()

if __name__ == '__main__':
	__main__()
	