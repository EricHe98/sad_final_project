from MultVAE_Dataset import *
from MultVAE_model import *
from MultVAE_training_helper import *
from torch import nn
from torch.utils.data import DataLoader
import datetime as dt
import argparse
import mlflow
import mlflow.pytorch

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
parser.add_argument('-d', 
                    '--dict_path', 
                    nargs = '?',
                    type = str,
                    help = 'Dictionary path',
                    default = '/scratch/work/js11133/sad_data/processed/hotel_hash.json')
parser.add_argument('-s', 
                    '--save_path', 
                    nargs = '?',
                    type = str,
                    help = 'models save path',
                    default = '/scratch/work/js11133/sad_data/models/multVAE/')
parser.add_argument('-l', 
                    '--num_layers', 
                    nargs = '?',
                    type = int,
                    help = 'Number of hidden layers in MultVAE encoder and Decoder',
                    default = 1)
parser.add_argument('-lr', 
                    '--learning_rate', 
                    nargs = '?',
                    type = float,
                    help = 'Learning Rate for model',
                    default = 1e-3)
parser.add_argument('-hd', 
                    '--hidden_dim', 
                    nargs = '?',
                    type = int,
                    help = 'Size of Hidden Dimension',
                    default = 600)
parser.add_argument('-lt', 
                    '--latent_dim', 
                    nargs = '?',
                    type = int,
                    help = 'Size of Latent Dimension',
                    default = 200)
args = parser.parse_args()

if __name__ == '__main__':
    
    #Define loaders
    train_loader, hotel_length = make_dataloader(data_path = args.train_path,
                                                hotel_path=args.dict_path,
                                                batch_size = 256)

    val_loader, _ = make_dataloader(data_path = args.val_path,
                                   hotel_path=args.dict_path,
                                   batch_size = 256)
    with mlflow.start_run(run_name = 'MultVAE'): 
      
      run_id = mlflow.active_run().info.run_id
      print('MLFlow Run ID is :{}'.format(run_id))
      mlflow.log_param('dataset', 'full')
      mlflow.log_param('train_split', 'train')
      mlflow.log_param('model_name', 'MultVAE')
      mlflow.log_param('run_id', run_id)

      if torch.cuda.is_available():
          device = torch.device("cuda")
          print('There are %d GPU(s) available.' % torch.cuda.device_count())
          print('We will use the GPU:', torch.cuda.get_device_name(0))
      else:
          print('No GPU available, using the CPU instead.')
          device = torch.device("cpu")    

      
      mlflow.log_param('device', device)
      mlflow.log_param('hotel_dim', hotel_length)
      mlflow.log_param('hidden_dim', args.hidden_dim)
      mlflow.log_param('latent_dim', args.latent_dim)
      mlflow.log_param('dropout', 0.5)
      mlflow.log_param('beta', 'annealed_to halfway')
      mlflow.log_param('learning_rate', args.learning_rate)
      mlflow.log_param('n_enc_hidden_layers', args.num_layers)
      mlflow.log_param('n_dec_hidden_layers', args.num_layers)

      # train, validate ..
      model = MultVae(item_dim=hotel_length,
                      hidden_dim=args.hidden_dim,
                      latent_dim=args.latent_dim,
                      n_enc_hidden_layers = args.num_layers,
                      n_dec_hidden_layers = args.num_layers,
                      dropout = 0.5
                     )
      model.to(device)
      time_start = dt.datetime.now()

      metrics, final_epoch =train_and_validate(
                                                model=model,
                                                train_loader=train_loader,
                                                valid_loader=val_loader,
                                                device = device,
                                                start_beta = 0.0,
                                                max_beta=1.0,
                                                num_epoch=400,
                                                learning_rate=args.learning_rate,
                                                max_patience=5,
                                                run_id = run_id,
                                                save_path = args.save_path,
                                              )
      time_end = dt.datetime.now()
      train_time = (time_end - time_start).total_seconds()


      with open('checkpoints/metrics_{}.pkl'.format(run_id), "wb" ) as f:
        pickle.dump(metrics,f)

      #mlflow.log_artifacts('/scratch/work/js11133/sad_data/models/multVAE', artifact_path = 'models_per_epoch')
      mlflow.log_artifact('checkpoints/metrics_{}.pkl'.format(run_id))
      
      mlflow.log_metric('Num_epochs', final_epoch + 1)
      mlflow.log_metric('training_time', train_time)
      print('Model trained in {} seconds'.format(train_time))

      mlflow.pytorch.save_model(pytorch_model = model, path = args.save_path + 'multvae_anneal_{}.uri'.format(run_id))



    

    
    
