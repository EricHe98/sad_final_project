from MultVAE_Dataset import *
from multVAE_model import *
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
def make_dataloader(data_path = None, hotel_path = None, batch_size = 256):
    hotel_dataset = BasicHotelDataset(data_path, hotel_path)
    hotel_length = hotel_dataset.hotel_length
    return DataLoader(hotel_dataset, batch_size = batch_size), hotel_length

def train(model,
        beta, 
        train_loader,
        optimizer,
        device):
    loss_per_epoch = 0
    bce_per_epoch = 0
    kld_per_epoch = 0
    model.train() 

    for data in train_loader: 
        #Send to devices
        x = data.to(device)
        # Foward pass thru model
        x_hat, mu, logvar = model(x)
        # Zero out optimizer gradients
        optimizer.zero_grad()
        # Loss and calculate gradients
        loss, bce, kld = VAE_loss_function(x_hat, x, mu, logvar, beta)
        # Backward Pass
        loss.backward()
        # Take the gradient descent step
        optimizer.step()
        #Record Loss
        loss_per_epoch += loss.item()
        bce_per_epoch += bce.item()
        kld_per_epoch += kld.item()

    train_loss = loss_per_epoch / len(train_loader.dataset)
    train_bce = bce_per_epoch / len(train_loader.dataset)
    train_kld = kld_per_epoch / len(train_loader.dataset)
    print('Train Loss: {:.6f}'.format(train_loss))
    return train_loss,train_bce,train_kld


def validate(model,
            beta,
            valid_loader,
            best_val_loss,
            device,
            save_path='checkpoints/multvae_basic_model.pth'):
    total_loss = 0
    model.eval()
    loss_per_epoch = 0
    bce_per_epoch = 0
    kld_per_epoch = 0
    with torch.no_grad():
        for data in valid_loader:
            x = data.to(device)
            x_hat, mu, logvar = model(x)
            loss, bce, kld = VAE_loss_function(x_hat, x, mu, logvar, beta)
            loss_per_epoch += loss.item()
            bce_per_epoch += bce.item()
            kld_per_epoch += kld.item()

        val_loss = loss_per_epoch / len(valid_loader.dataset) 
        val_bce = bce_per_epoch / len(valid_loader.dataset)
        val_kld = kld_per_epoch / len(valid_loader.dataset)
        print('Validation Loss: {:.6f}'.format(val_loss))

    #Something here for validation ndcg@100?

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print('Saved best model in the checkpoint directory\n')
    return val_loss, best_val_loss, val_bce, val_kld

def train_and_validate(model,
                       train_loader,
                       valid_loader,
                       device,
                       beta = 1.0,
                       num_epoch = 100,
                       learning_rate = 1e-4,
                       log_interval = 1,
                       max_patience = 5,
                       metrics_file_path = 'checkpoints/metrics.pkl',
                       ):
    #Initialize stuff
    patience_counter = 0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    train_loss_history = []
    train_bce_history = []
    train_kld_history = []
    
    val_loss_history = []
    val_bce_history = []
    val_kld_history = []
    best_val_loss = 10e7

    
    for epoch_ii  in range(num_epoch):
        print("Epoch {}".format(epoch_ii + 1,))
        print("Starting Time for Epoch {}:{}".format(epoch_ii + 1, datetime.now()))

        #Train
        train_loss,train_bce,train_kld = train(model,beta,train_loader,optimizer, device)
        train_loss_history.append(train_loss)
        train_bce_history.append(train_bce)
        train_kld_history.append(train_kld)
          
        
        # Validate
        current_val_loss, new_best_val_loss,val_bce,val_kld = validate(model,beta,valid_loader, best_val_loss, device)
        val_loss_history.append(current_val_loss)
        val_bce_history.append(val_bce)
        val_kld_history.append(val_kld)
        if current_val_loss >= best_val_loss:
            patience_counter+=1
        else:
            patience_counter=0
        best_val_loss=new_best_val_loss
        print('patience',patience_counter)
        if patience_counter>max_patience:
             break
        print("Starting Time for Epoch {}:{}".format(epoch_ii + 1, datetime.now()))
        #End For
    
    metrics= (train_loss_history,train_bce_history,train_kld_history, val_loss_history,val_bce_history,val_kld_history)
    metrics_file_path = metrics_file_path
    with open(metrics_file_path, "wb" ) as f:
        pickle.dump(metrics,f)
    return train_loss_history,train_bce_history,train_kld_history, val_loss_history,val_bce_history,val_kld_history

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
    train_loader, hotel_length = make_dataloader(data_path = args.train_path,
                                                hotel_path=args.dict_path,
                                                batch_size = 256)

    val_loader, _ = make_dataloader(data_path = args.val_path,
                                   hotel_path=args.dict_path,
                                   batch_size = 256)

    test_loader, _ = make_dataloader(data_path = args.test_path,
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
    
    train_and_validate(model=model,
                       train_loader=train_loader,
                       valid_loader=test_loader,
                       device = device,
                       beta=1.0,
                       num_epoch=2,
                       learning_rate=1e-4,
                       log_interval=1,
                       max_patience=5,
                       metrics_file_path='checkpoints/metrics.pkl',
                       )



    

    
    