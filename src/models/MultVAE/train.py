from MultVAE_Dataset import *
from MultVAE_model import *
from MultVae_training_helper import * as helper 
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
import argparse


"""
TODO:
    1.) Probabilities to 0s and 1s
    2.) How to rank and calculate ndcg@100/recall@100
    3.) Put into ML Flow

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File Paths for training, validating, and testing')
    parser.add_argument('-tr', 
                        '--train_path', 
                        nargs = '?',
                        type = str, 
                        help = 'training data path',
                        default = '/scratch/work/js11133/sad_data/processed/full/train/user_to_queries.pkl')
    parser.add_argument('-v',
                        '--val_path', 
                        nargs = '?',
                        type = str,
                        help = 'validation data path',
                        default ='/scratch/work/js11133/sad_data/processed/full/val/user_to_queries.pkl' )
    parser.add_argument('-t', 
                        '--test_path', 
                        nargs = '?',
                        type = str, 
                        help = 'testing data path',
                        default = '/scratch/work/js11133/sad_data/processed/full/test/user_to_queries.pkl')
    parser.add_argument('-d', 
                        '--dict_path', 
                        nargs = '?',
                        type = str,
                        help = 'Dictionary path',
                        default = '/scratch/work/js11133/sad_data/processed/hotel_hash.json')
    args = parser.parse_args()
    
    print('Torch Version: {}'.format(torch.__version__))
    
    #Define loaders
    train_loader, hotel_length = helper.make_dataloader(data_path = args.train_path,
                                                hotel_path=args.dict_path,
                                                batch_size = 256)

    val_loader, _ = helper.make_dataloader(data_path = args.val_path,
                                   hotel_path=args.dict_path,
                                   batch_size = 256)

    test_loader, _ = helper.make_dataloader(data_path = args.test_path,
                                   hotel_path=args.dict_path,
                                   batch_size = 256)
    
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")    
    
    # train, validate ..
    
    zdim=20 
    model = MultVae(item_dim=hotel_length,
                    hidden_dim=600,
                    latent_dim=200,
                    n_enc_hidden_layers = 1,
                    n_dec_hidden_layers = 1,
                    dropout = 0.5
                   )
    
    model.to(device)
    
    helper.train_and_validate(model=model,
                       train_loader=train_loader,
                       valid_loader=test_loader,
                       device = device,
                       beta=1.0,
                       num_epoch=400,
                       learning_rate=1e-4,
                       log_interval=1,
                       max_patience=5,
                       metrics_file_path='checkpoints/metrics.pkl',
                       )



    

    
    
