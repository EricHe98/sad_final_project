from MultVAE_Dataset import *
from multVAE_model import *
from torch import nn
from torch.utils.data import DataLoader

def make_dataloader(data_path = None, hotel_path = None):
    hotel_dataset = BasicHotelDataset(data_path, hotel_path)
    return DataLoader(hotel_dataset, batch_size = 10)

if __name__ == '__main__':
    train_loader = make_dataloader(data_path = 'data/processed/user_to_queries.pkl', hotel_path ='data/processed/hotel_hash.json')
    print(next(iter(train_loader)))
    