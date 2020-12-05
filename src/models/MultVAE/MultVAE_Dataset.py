import pickle
from scipy import sparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
class BasicHotelDataset(Dataset):

    def __init__(self, data_path = None, dict_path = None):
        """
        Args
            data_path (string): Path to the pkl file
            dict_path: Path to the hotel hashes
        """
        if data_path is None:
            raise ValueError('Please specify data_path')
        elif os.path.isfile(data_path) == False:
            raise ValueError('Not Correct file path')
        
        if dict_path is None:
            raise ValueError('Need path of hashes')
        elif os.path.isfile(dict_path) == False:
            raise ValueError('Not Correct file path to hashes')
                
        with open(data_path,'rb') as fp:
            self.data = pickle.load(fp)
        
        self.data = {key: value[1] for (key, value) in self.data.items()}
        
        num_keys = len(self.data.keys())
        dataset_keys = self.data.keys()
        
        self.idx_to_dataset_keys_dict = dict(zip(range(num_keys),dataset_keys))
        
        with open(dict_path, 'r') as fp:
            self.hotel_length = len(json.load(fp))
        
    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist

        if isinstance(idx, int):
            idx = [idx]
        
        user_interactions = [self.data[self.idx_to_dataset_keys_dict[k]] for k in idx] #list of dicts
        sparse_dok = sparse.dok_matrix((len(idx),self.hotel_length),dtype=np.float32)
        for i in range(len(user_interactions)):
            for j in user_interactions[i].keys():
                sparse_dok[i,j] = user_interactions[i][j]
           
        return torch.tensor(sparse_dok.toarray())