import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import sparse
import json
import random
import torch
from torch.utils.data import Dataset
import torch.sparse
from preprocessing import *
import pickle
class HotelDataset(Dataset):

    def __init__(self, data_path = None, dict_path = None):
        """
        Args

            data_path (string): Path to the csv file
        """
        if data_path is None:
            raise ValueError('Please specify data_path')
        if dict_path is None:
            raise ValueError('Need path of hashes')
        
        _ , ext = os.path.splitext(data_path)
        if ext != 'csv':
            raise ValueError('Incorrect File to upload')
        _, ext2 = os.path.splitext(dict_path)
        if ext2 != 'json':
            raise ValueError('Incorrect File to use as indicies')
        self.data_sparse = pd.read_csv(data_path)


        with open(os.path.join(dict_path, 'user_hash.json'), 'r') as fp:
            self.user_id_indexed = json.load(fp)

        with open(os.path.join(dict_path, 'hotel_hash.json'), 'r') as fp:
            self.hotel_id_indexed = json.load(fp)

        self.sparse_index = torch.LongTensor([[data_sparse.user_id, data_sparse.hotel_id]])
        self.sparse_value = torch.LongTensor([[data_sparse.labels]])

        self.interactions = torch.sparse.FloatTensor(sparse_index,
                                                    sparse_value,
                                                    torch.size([len(user_id_indexed), len(hotel_id_indexed)]
                                                    )).to_dense()
        def __len__(self):
            return len(self.data_sparse)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist
            return self.interactions[idx, :]




def read_args(input_list):
    if len(input_list) != 3:
        raise ValueError("Run with arguments data_path, output_path")
    
    if os.path.isdir(input_list[1]):
        data_path = input_list[1]
    else:
        raise ValueError("Incorrect Data Directory, please give Directory path")
    
    if os.path.exists(input_list[2]):
        output_path = input_list[2]
    else:
        raise ValueError("Path for output file does not exist")

    return data_path, output_path

if __name__ == "__main__":
    
    (data_path, output_path) = read_args(sys.argv)
    
    # Step 0, Read in Data
    columns_to_read = ['search_result_id','search_request_id', 'hotel_id', 'user_id','label', 'check_in', 'check_out',
       'reward_program_hash', 'advance_purchase_days',
       'number_of_nights', 'number_of_rooms', 'number_of_adults',
       'srq_latitude', 'srq_longitude', 'check_in_weekday',
       'check_out_weekday', 'srq_weekhour', 'weekday_travel',
       'weekend_travel']

  
    data = read_parquet(data_path, 
                    num_partitions = None,
                    randomize = False,
                    verbose = True,
                    columns = columns_to_read)

    
    # Step 1, basic conversions/drop NaNs
    print('Starting Step 1')
    user_hash_json_path = os.path.join(output_path, 'user_hash.json')
    hotel_hash_json_path = os.path.join(output_path, 'hotel_hash.json')
    
    data = df_conversions(data, user_hash_json_path, hotel_hash_json_path)
    
    
    
    # Step 2, create dict of user_id -> query_struct
    print('Starting Step 2')

    user_id_to_query_struct_dict = create_user_id_to_query_struct_dict(data)
    with open(os.path.join(output_path, 'user_to_queries.pkl'), 'wb') as fp:
        pickle.dump(user_id_to_query_struct_dict, fp)

    # Step 3, Create context_df and encoder vector
    print('Starting Step 3')
    cat_vars_to_use = ['reward_program_hash',
                    'check_in_weekday',
                    'check_out_weekday',
                    'weekday_travel',
                    'weekend_travel']
    context_df, cat_onehot_enc = create_context_df_and_cat_encoder(data, cat_vars_to_use)
    
    context_df.to_csv(os.path.join(output_path, 'context_df.csv'))
    
    with open(os.path.join(output_path, 'cat_onehot_enc.pkl'), 'wb') as fp:
        pickle.dump(cat_onehot_enc, fp)
    



    








