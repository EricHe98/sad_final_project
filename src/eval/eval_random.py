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
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to predict on (small_100, small_all, all)')
parser.add_argument('split', choices=['train', 'val', 'test'],
	help='which split of the dataset to predict on (train, val, test)')

args = parser.parse_args()
data_path = 'data/raw'

def __main__():
	data = read_parquet(os.path.join(data_path, args.dataset, args.split),
		columns=['hotel_id', 'user_id', 
			'search_request_id', 'hotel_cumulative_share',
			'display_rank', 'price_rank', 'label'])

	ndcgs_random = []
	for i in range(50):
		data['randn'] = np.random.randn(len(data))

		ndcgs_random.append(lm.ndcg(
		    data, 
		    groupby='search_request_id',
		    ranker='randn')\
			.mean())

	print(np.mean(ndcgs_random))

if __name__ == '__main__':
	__main__()
