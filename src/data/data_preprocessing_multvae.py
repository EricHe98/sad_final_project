import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import sparse
import json
import random
from torch.utils.data import Dataset
import torch.sparse

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



def read_parquet(data_path, num_partitions: None, randomize = True, verbose = True, columns = ['hotel_id', 'user_id','label']):
    files = os.listdir(data_path)
    if randomize:
        random.shuffle(files)
    
    if num_partitions is None:
        num_partitions = len(files)
    
    data = []
    num_reads = 0
    for file_path in files:
        if num_reads >= num_partitions:
            if verbose:
                print('Finished reading {} .parquet Files'.format(num_partitions))
            break
        
        _ , ext = os.path.splitext(file_path)
        
        if ext == '.parquet':
            fp = os.path.join(data_path, file_path)
            data.append(pd.read_parquet(os.path.join(data_path, file_path), columns = columns))
            
            if verbose:
                print('Reading in data of shape {} from {}'.format(data[-1].shape, fp))
            
            num_reads += 1
        else: 
            continue
    data = pd.concat(data, axis=0)
    
    if verbose:
        print('Total dataframe of shape {}'.format(data.shape))
    
    return data

def read_args(input_list):
    if len(input_list) != 5:
        raise ValueError("Run with arguments data_path, output_path")
    
    if os.path.isdir(input_list[1]):
        data_path = input_list[1]
    else:
        raise ValueError("Incorrect Data Directory, please give Directory path")
    
    if os.path.exists(input_list[2]):
        output_path = input_list[2]
    else:
        raise ValueError("Path for output file does not exist")

    make_dict = True if input_list[3] == 'True' else False

    if input_list[4] == 'None':
        train_val_test = None
    elif input_list[4] == 'train' or input_list[4] == 'val' or input_list[4] == 'test':
        train_val_test = input_list[3]
    else:
        raise ValueError("Specify train, test or val")

    return data_path, output_path, make_dict, train_val_test

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description= 'Process Set for input into MultVAE')
    #parser.add_argument('--data_path', 
    #                    type = str, 
    #                    default = None,
    #                    help = 'Directory Location of dataset to be processed')
    #parser.add_argument('--output_path',
    #                    type = str,
    #                    default = None,
    #                    help = 'Location to save processed data')
    #parser.add_argument('--make_dict',
    #                    type = bool,
    #                    default = False,
    #                    help = 'Make hash tables for users and hotels')
    #parser.add_argument('--train_val_test',
    #                    type = str,
    #                    default = 'train',
    #                    help = 'which dataset to process')
    #args = parser.parse_args()
    (data_path, output_path, make_dict, train_val_test) = read_args(sys.argv)

    if make_dict:
        data = read_parquet(data_path, 
                num_partitions = None,
                randomize = False,
                verbose = True,
                columns = ['hotel_id','user_id','label'])

        unique_users = data.user_id.unique()
        unique_hotels = data.hotel_id.unique() 

        user_id_indexed = dict((user_id, i) for i, user_id in enumerate(unique_users))
        hotel_id_indexed = dict((hotel_id, i) for i,hotel_id in enumerate(unique_hotels))
        user_hashed = data.user_id.apply(lambda x: user_id_indexed[x])
        hotel_hashed = data.hotel_id.apply(lambda x: hotel_id_indexed[x])

        with open(os.path.join(output_path, 'user_hash.json'), 'w') as fp:
            json.dump({str(key): value for key, value in user_id_indexed.items()},
                fp, 
                sort_keys=True)

        with open(os.path.join(output_path, 'hotel_hash.json'), 'w') as fp:
            json.dump({str(key): value for key, value in hotel_id_indexed.items()},
                fp,
                sort_keys=True)
        print('Finished hashing indicies')
    
    else:
        data = read_parquet(data_path, 
                        num_partitions = None,
                        randomize = False,
                        verbose = True,
                        columns = ['search_result_id','search_request_id', 'hotel_id',
                                    'user_id','label', 'check_in', 'check_out',
                                    'reward_program_hash', 'advance_purchase_days',
                                    'number_of_nights', 'number_of_rooms', 'number_of_adults',
                                    'srq_latitude', 'srq_longitude', 'check_in_weekday',
                                    'check_out_weekday', 'srq_weekhour', 'weekday_travel',
                                    'weekend_travel'])
        data.check_in = pd.to_datetime(df['check_in'],yearfirst=True)
        data.check_out = pd.to_datetime(df['check_out'],yearfirst=True)

    data = data.dropna(subset = ['user_id'])
    data.user_id = data.user_id - 1e10

    with open(os.path.join(output_path, 'user_hash.json'), 'r') as fp:
        user_id_indexed = json.load(fp)

    with open(os.path.join(output_path, 'hotel_hash.json'), 'r') as fp:
        hotel_id_indexed = json.load(fp)

    user_hashed = data.user_id.apply(lambda x: user_id_indexed[x])
    hotel_hashed = data.hotel_id.apply(lambda x: hotel_id_indexed[x])
    
    final_reindexed_data = pd.DataFrame(data = {'user_id' : user_hashed, 
                                                'hotel_id':hotel_hashed, 
                                                'label': data.label},
                                        columns = ['user_id', 'hotel_id', 'label'] )

    final_reindexed_data.to_csv(os.path.join(output_path, 'final_reindex_{}.csv'.format(train_val_test)), index = False)
    print('Finished Reindexing {} Data'.format(train_val_test))

















