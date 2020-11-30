import pickle
from scipy import sparse
import json
import os
import sys

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
        if dict_path is None:
            raise ValueError('Need path of hashes')
        
        #_ , ext = os.path.splitext(data_path)
        
        #if ext != 'pkl':
         #   raise ValueError('Incorrect File to upload')
        
        #_, ext2 = os.path.splitext(dict_path)
        
        #if ext2 != 'json':
         #   raise ValueError('Incorrect File to use as indicies')
        
        with open(data_path,'rb') as fp:
            self.data = pickle.load(fp)
        
        self.data = {key: value[1] for (key, value) in self.data.items()}

        with open(dict_path, 'r') as fp:
            self.hotel_length = len(json.load(fp))
        
    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist

        user_interactions = [self.data[k] for k in idx] #list of dicts
        sparse_dok = sparse.dok_matrix((len(idx),self.hotel_length))
        for i in range(len(user_interactions)):
            for j in user_interactions[i].keys():
                sparse_dok[i,j] = user_interactions[i][j]
           
        return torch.tensor(sparse_dok.toarray())