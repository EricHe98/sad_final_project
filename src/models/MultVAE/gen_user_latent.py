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
from scipy import sparse

import argparse
parser = argparse.ArgumentParser(description='Use MultVAE model to predict on validation set.')

parser.add_argument('-m',
                    '--model_folder', 
                    type = str,
                    required=True,
                    help='model_folder. should be a dir(.uri). Needs to be MultVAE class.',
                    default = '/scratch/abh466/sad_data/model/MultVAE/'
                    )
parser.add_argument('-n',
                    '--model_run_id', 
                    type = str,
                    required=True,
                    help='model_run_id. should be the run_id of all the models in the model_folder',
                    )
                    
parser.add_argument('-e',
                    '--epoch', 
                    type = int,
                    required=True,
                    help='max epoch, the last epoch that you want to validate towards',
                    default = 0
                    )
                    
parser.add_argument('-d',
                    '--dataset_pkl', 
                    nargs = '?',
                    type = str,
                    help='dataset pkl. Should be a user_to_queries.pkl. Check preprocessing.py for info on that structure',
                    default ='/scratch/abh466/sad_data/processed/full/val/user_to_queries.pkl' )

parser.add_argument('-i',
                    '--hotel_hash', 
                    nargs = '?',
                    type = str,
                    help='hotel_hash.json. Check make_hashes.py for info on the hash',
                    default ='/scratch/abh466/sad_data/processed/hotel_hash.json')

parser.add_argument('-u',
                    '--user_hash', 
                    nargs = '?',
                    type = str,
                    help='user_hash.json. Check make_hashes.py for info on the hash',
                    default ='/scratch/abh466/sad_data/processed/user_hash.json')
                    
parser.add_argument('-o',
                    '--output_dir', 
                    nargs = '?',
                    type = str,
                    help='output directory where embeddings will go',
                   )

args = parser.parse_args()

def densify_sparse_vec(user_interaction_dict, hotel_length):
    sparse_dok = sparse.dok_matrix((1,hotel_length),dtype=np.float32)
    sparse_obs = sparse.dok_matrix((1,hotel_length),dtype=np.float32)
    for j in user_interaction_dict.keys():
        sparse_dok[0,j] = user_interaction_dict[j]
        sparse_obs[0,j] = 1
    return torch.tensor(sparse_dok.toarray()),torch.tensor(sparse_obs.toarray())


def gen_user_emb(
                model_folder,
                model_run_id,
                epoch,
                dataset_pkl_path ,
                hotel_hash ,
                user_hash ,
                output_dir
                ):
   # if torch.cuda.is_available():

   #     device = torch.device("cuda")
   #     print('There are %d GPU(s) available.' % torch.cuda.device_count())
   #     print('We will use the GPU:', torch.cuda.get_device_name(0))
   # else:
   #     print('No GPU available, using the CPU instead.')
    device = torch.device("cpu") 

    with open(dataset_pkl_path,'rb') as f:
        user_to_query_struct = pickle.load(f)

    with open(user_hash, 'r') as fp:
        user_id_indexed = json.load(fp) 
    
    with open(hotel_hash, 'r') as fp:
        hotel_length = len(json.load(fp)) 
        
    #invert the maps so we can go back to user_id
    user_idx_to_user_id = {v: k for k, v in user_id_indexed.items()}
    
    model_name = 'multvae_{0}_annealed_epoch_{1}.uri'.format(model_run_id,str(int(epoch)))
    model_path = os.path.join(model_folder,model_name)

    model = mlflow.pytorch.load_model(model_path)
    print('Loaded model from ',model_path) 
    model.to(device)
    model.eval()
    print('loading done')
    #pd.DataFrame(df[0].values.tolist())
    lat_df = pd.DataFrame(columns=['user_id', 'latent'])

    for user in user_to_query_struct.keys():
        x, _ = densify_sparse_vec(user_to_query_struct[user][1], hotel_length)
        x.unsqueeze(dim = 0).to(device)
        emb, logvar  = model.encoder(x)
        user_id = user_idx_to_user_id[user]
 
        lat_df = lat_df.append({'user_id':user_id, 'latent':list(map(float,list(emb.cpu().detach().squeeze())))}, ignore_index = True)

    lat_df = pd.concat([lat_df.user_id, pd.DataFrame(lat_df.latent.values.tolist())], axis = 1)   
    columns = ['user_id']
    latents = ['latent_{}'.format(i) for i in range(200)]
    for i in latents:
        columns.append(i)
    lat_df.columns = columns
    return lat_df

if __name__ == '__main__':
    model_folder= args.model_folder
    model_run_id= args.model_run_id
    epoch = args.epoch
    dataset_pkl_path = args.dataset_pkl
    hotel_hash = args.hotel_hash
    user_hash = args.user_hash
    output_dir = args.output_dir
    
    lat_df = gen_user_emb(
                model_folder =model_folder,
                model_run_id=model_run_id,
                epoch=epoch,
                dataset_pkl_path = dataset_pkl_path,
                hotel_hash = hotel_hash,
                user_hash = user_hash,
                output_dir = output_dir
               )
    lat_df.to_parquet(output_dir+'train_emb.parquet')
