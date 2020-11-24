import os
import sys
import pandas as pd
import numpy as np
from scipy import sparse
import json
import random
from torch.utils.data import Dataset
import torch.sparse

class HotelDataset(Dataset):

    def __init__(self, data_path = None):
        """
        Args

            data_path (string): Path to the csv file
        """
        if data_path is None:
            raise ValueError('Please specify data_path')
        _ , ext = os.path.splitext(data_path)
        
        if ext != 'csv':
            raise ValueError('Incorrect File to upload')
        self.data_sparse = pd.read_csv(data_path)
        self.sparse_index = torch.LongTensor([[data_sparse.user_id, data_sparse.hotel_id]])
        self.sparse_value = torch.LongTensor([[data_sparse.labels]])
        self.interactions = torch.sparse.FloatTensor(sparse_index,
                                                            sparse_value,
                                                            torch.size([len(data_sparse.user_id), len(data_sparse.hotel_id)]
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
    if len(input_list) != 4:
        raise ValueError("Run with arguments data_path, output_path")
    
    if os.path.isdir(input_list[1]):
        data_path = input_list[1]
    else:
        raise ValueError("Incorrect Data Directory, please give Directory path")
    
    if os.path.exists(input_list[2]):
        output_path = input_list[2]
    else:
        raise ValueError("Path for output file does not exist")

    if input_list[3] == 'train' or input_list[3] == 'val' or input_list[3] == 'test':
        train_val_test = input_list[3]
    else:
        raise ValueError("Specify train, test or val")

    return data_path, output_path, train_val_test

if __name__ == "__main__":
    (data_path, output_path, train_val_test) = read_args(sys.argv)
    data = read_parquet(data_path, 
                        num_partitions = None,
                        randomize = False,
                        verbose = True,
                        columns = ['hotel_id', 'user_id','label'])

    data = data.dropna(subset = ['user_id'])
    data.user_id = data.user_id - 10000000000

    if train_val_test == 'train':
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

    else: 
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

















