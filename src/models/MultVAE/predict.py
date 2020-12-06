import os
import json
import pickle
import sys
import traceback
import datetime as dt

import numpy as np 
import pandas as pd

from  src.data.preprocessing import get_user_entire_interaction_vec

import mlflow

import argparse
parser = argparse.ArgumentParser(description='Use MultVAE model to predict on validation set.')
parser.add_argument('-r',
                    '--run_id',
                    type=int,
                    required=True,
                    ) 
parser.add_argument('-m',
                    '--multvae_model', 
                    type = str,
		    required=True,
                    help='multVAE model. Should be a pytorch checkpoint (.pth file). Needs to be MultVAE class.',
                    )
                    
parser.add_argument('-d',
                    '--dataset_pkl', 
                    nargs = '?',
                    type = str,
                    help='dataset pkl. Should be a user_to_queries.pkl. Check preprocessing.py for info on that structure',
                    default ='/scratch/work/js11133/sad_data/processed/full/val/user_to_queries.pkl' )
parser.add_argument('-h',
                    '--hotel_hash', 
                    nargs = '?',
                    type = str,
                    help='hotel_hash.json. Check make_hashes.py for info on the hash',
                    default ='/scratch/work/js11133/sad_data/processed/hotel_hash.json' )
                    
parser.add_argument('-o',
                    '--output_dir', 
                    nargs = '?',
                    type = str,
                    help='output directory where predictions will go',
                   )

args = parser.parse_args()


def __main__():
    mlflow.start_run(run_id=args.run_id)
    # Load user to query_struct
    with open(args.dataset_pkl,'rb') as f:
        user_to_query_struct = pickle.load(f)
    # Load hotel_id to index dictionary
    with open(hotel_hash_json_path, 'r') as fp:
        hotel_id_indexed = json.load(fp)
    #invert the map so we can go back to hotel_id
    hotel_idx_to_hotel_id = {v: k for k, v in my_map.items()}
    
    # Get all the user interaction vectors
    all_user_ids = user_to_query_struct[1].keys()
    X = [get_user_entire_interaction_vec(user_id) for user_id in all_user_ids]
    print(X.shape)
    
  
    # Load our multVAE model
    model = mlflow.pytorch.load_model('runs:/{}/model.json'.format(args.run_id))
    
    # generate predictions
    y = model(X)
    
    #pred_df = 
    
    raise NotImplementedError
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
        '''
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
         '''
   
if __name__ == '__main__':
	__main__()
