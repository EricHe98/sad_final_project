import os
import json
import pickle
import sys
import traceback
import datetime as dt

import numpy as np 
import pandas as pd

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import Dataset
from MultVAE_Dataset import BasicHotelDataset
from scipy import sparse
import src.modules.letor_metrics as lm

import argparse
parser = argparse.ArgumentParser(description='Use MultVAE model to predict on validation set.')

parser.add_argument('-m',
                    '--model_folder', 
                    type = str,
                    required=True,
                    help='model_folder. should be a dir. Needs to be MultVAE class.',
                    )
parser.add_argument('-n',
                    '--model_run_id', 
                    type = str,
                    required=True,
                    help='model_run_id. should be the run_id of all the models in the model_folder',
                    )
                    
parser.add_argument('-e',
                    '--max_epoch', 
                    type = int,
                    required=True,
                    help='max epoch, the last epoch that you want to validate towards',
                    )
                    
parser.add_argument('-d',
                    '--dataset_pkl', 
                    nargs = '?',
                    type = str,
                    help='dataset pkl. Should be a user_to_queries.pkl. Check preprocessing.py for info on that structure',
                    default ='/scratch/work/js11133/sad_data/processed/full/val/user_to_queries.pkl' )

parser.add_argument('-i',
                    '--hotel_hash', 
                    nargs = '?',
                    type = str,
                    help='hotel_hash.json. Check make_hashes.py for info on the hash',
                    default ='/scratch/work/js11133/sad_data/processed/hotel_hash.json')

parser.add_argument('-u',
                    '--user_hash', 
                    nargs = '?',
                    type = str,
                    help='user_hash.json. Check make_hashes.py for info on the hash',
                    default ='/scratch/work/js11133/sad_data/processed/user_hash.json')
                    
parser.add_argument('-o',
                    '--output_dir', 
                    nargs = '?',
                    type = str,
                    help='output directory where predictions will go',
                   )

args = parser.parse_args()


def get_single_query_interaction_vec(user_id_to_query_struct_dict,user_id,sr_id):
    return user_id_to_query_struct_dict[user_id][0][sr_id]

def get_user_entire_interaction_vec(user_id_to_query_struct_dict,user_id):
    return user_id_to_query_struct_dict[user_id][1]

def densify_sparse_vec(user_interaction_dict, hotel_length):
    sparse_dok = sparse.dok_matrix((1,hotel_length),dtype=np.float32)
    sparse_obs = sparse.dok_matrix((1,hotel_length),dtype=np.float32)
    for j in user_interaction_dict.keys():
        sparse_dok[0,j] = user_interaction_dict[j]
        sparse_obs[0,j] = 1
    return torch.tensor(sparse_dok.toarray()),torch.tensor(sparse_obs.toarray())

def val_metrics(model_folder,
                model_run_id,
                max_epoch,
                dataset_pkl_path ,
                hotel_hash ,
                user_hash ,
                output_dir
               ):
    print('IN MAIN') 
    #Check for CUDA
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu") 
    # Load user to query_struct
    with open(dataset_pkl_path,'rb') as f:
        user_to_query_struct = pickle.load(f)
    #Put the dataset into the dataloader
    dataset = BasicHotelDataset(data_path = dataset_pkl_path, dict_path = hotel_hash)

    
    #Create sr_id_to_user_id dictionary
    sr_id_to_user_id_hashed = {}
    for user_id_hashed in user_to_query_struct.keys():
        sr_ids = user_to_query_struct[user_id_hashed][0].keys()
        for sr_id in sr_ids:
            sr_id_to_user_id_hashed[sr_id] = user_id_hashed
    
    
    # Load hotel_id to index dictionary
    with open(hotel_hash, 'r') as fp:
        hotel_id_indexed = json.load(fp)
    
    # Load user_id to index dictionary
    with open(user_hash, 'r') as fp:

        user_id_indexed = json.load(fp)  
        
    #invert the maps so we can go back to hotel_id and user_id
    user_idx_to_user_id = {v: k for k, v in user_id_indexed.items()}
    hotel_idx_to_hotel_id = {v: k for k, v in hotel_id_indexed.items()}
    
    
    # Get user_idx to/from user_id mappings 
    dlkeys_to_user_id = dataset.idx_to_dataset_keys_dict
    user_id_to_dlkeys = {v: k for k, v in dlkeys_to_user_id.items()}
    
    
    #Make model_path_list
    model_path_list =[]
    for epoch in range(max_epoch):
        model_name = 'multvae_{0}_epoch_{1}.uri'.format(model_run_id,str(int(640+epoch)))
        model_path = os.path.join(model_folder,model_name)
        model_path_list.append(model_path)
    #Now loop over each model we need to eval on
    ndcg_list = []
    for model_path in model_path_list:
        # Load our multVAE model
        model = mlflow.pytorch.load_model(model_path)
        print('Loaded model from ',model_path) 
        model.to(device)
        print('loading done')
        # generate predictions
        df_list = []


        for sr_id in sr_id_to_user_id_hashed.keys():
            user_id = sr_id_to_user_id_hashed[sr_id]
            user_id_unhashed = user_idx_to_user_id[user_id]

            # GET SINGLE QUERY, OR ENTIRE interaction?
            user_interaction_vec = get_single_query_interaction_vec(user_to_query_struct,user_id,sr_id)
            
            x, observed_vec = densify_sparse_vec(user_interaction_vec,dataset.hotel_length)
            label = x
            #print(x.shape)
            
            x = x.to(device)

            x_preds, mu, logvar = model(x.unsqueeze(dim=0))

            model.eval()

            x_preds = pd.DataFrame({'score':x_preds.cpu().detach().squeeze().numpy(),
                                    'observed':observed_vec.cpu().detach().squeeze().numpy(),
                                    'label':label.cpu().detach().squeeze().numpy()}
                                  ) 
            x_preds = x_preds[x_preds['observed']==1]
            x_preds['hotel_id'] = x_preds.index.map(hotel_idx_to_hotel_id.get)
            x_preds['search_request_id'] = sr_id
            x_preds['user_id'] = user_id_unhashed

            df_list.append(x_preds)
        print('end for loop')
        pred_array = pd.concat(df_list)   
        print('concat ended')
        pred_array['rank'] = pred_array\
            .groupby('search_request_id')\
            ['score']\
            .rank(ascending=False)
        #Now that we have preds, calculate NDCG
        ndcg_model = lm.ndcg(
            pred_array, 
            groupby='search_request_id',
            ranker='score')
        ndcg_score = ndcg_model.mean()
        print(ndcg_score)
        ndcg_list.append(ndcg_score)
    print(ndcg_list)
    ndcg_df = pd.DataFrame({'ndcg_score':ndcg_list})
    print(ndcg_df)
    ndcg_df.to_csv(os.path.join(output_dir,'ndcg_score_{}.csv'.format(model_run_id) ))
    
   

if __name__ == '__main__':
    model_folder= args.model_folder
    model_run_id= args.model_run_id
    max_epoch = args.max_epoch
    dataset_pkl_path = args.dataset_pkl
    hotel_hash = args.hotel_hash
    user_hash = args.user_hash
    output_dir = args.output_dir
    
    val_metrics(model_folder = model_folder,
                model_run_id=model_run_id,
                max_epoch=max_epoch,
                dataset_pkl_path = dataset_pkl_path,
                hotel_hash = hotel_hash,
                user_hash = user_hash,
                output_dir = output_dir
               )
