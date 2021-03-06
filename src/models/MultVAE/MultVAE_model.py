import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as numpy

class MultVAE_encoder(nn.Module):
    def __init__(self, 
                item_dim: int, 
                hidden_dim = 600, 
                latent_dim = 200, 
                n_hidden_layers = 1, 
                dropout = 0.5,
                nonlinearity = nn.Tanh):
        super(MultVAE_encoder, self).__init__()
        self.item_dim = item_dim
        self.latent_dim = latent_dim
        self.nonlinearity = nn.Tanh()
        self.layers = nn.Sequential()
        self.layers.add_module("input_dropout", nn.Dropout(dropout))
        self.layers.add_module("linear_enc_1",
                                nn.Linear(in_features = item_dim, out_features = hidden_dim))
        self.layers.add_module("Tanh_enc_1", self.nonlinearity)
        
        if n_hidden_layers > 0:
            for i in range(n_hidden_layers):
                self.layers.add_module('hidden_enc_{}'.format(i + 1),
                                        nn.Linear(in_features = hidden_dim, out_features = hidden_dim))
                self.layers.add_module("Tanh_enc_{}".format(i + 2), self.nonlinearity)
        
        self.mu = nn.Linear(in_features = hidden_dim, out_features = latent_dim)
        self.logvar = nn.Linear(in_features = hidden_dim, out_features = latent_dim)
    
    def forward(self, x):
        output = self.layers(x)
        mu = self.mu(output)
        logvar= self.logvar(output)

        return mu, logvar
        
class MultVAE_decoder(nn.Module):
    def __init__(self, item_dim, hidden_dim = 600, latent_dim = 200, n_hidden_layers = 1,nonlinearity=nn.Tanh):
        super(MultVAE_decoder, self).__init__()
        self.item_dim = item_dim
        self.latent_dim = latent_dim
        self.nonlinearity = nonlinearity()
        self.layers = nn.Sequential()
        self.layers.add_module("linear_dec_1",
                                nn.Linear(in_features = latent_dim, out_features = hidden_dim))
        self.layers.add_module("Tanh_dec_1", self.nonlinearity)
        
        if n_hidden_layers > 0:
            for i in range(n_hidden_layers):
                self.layers.add_module('hidden_dec_{}'.format(i + 1),
                                        nn.Linear(in_features = hidden_dim, out_features = hidden_dim))
                self.layers.add_module("Tanh_dec_{}".format(i + 2), self.nonlinearity)
        
        self.item_layer = nn.Linear(in_features = hidden_dim, out_features = item_dim)

    def forward(self, x):
        output = self.layers(x)
        items = self.item_layer(output)
        items = self.nonlinearity(items)
        return items


class MultVae(nn.Module):
    def __init__(self, item_dim, hidden_dim = 600, latent_dim = 200, n_enc_hidden_layers = 1, n_dec_hidden_layers = 1, dropout = 0.5):
        super(MultVae, self).__init__()
        self.item_dim = item_dim
        self.encoder = MultVAE_encoder(
                                        item_dim = item_dim,
                                        hidden_dim = hidden_dim, 
                                        latent_dim = latent_dim,
                                        n_hidden_layers = n_enc_hidden_layers,
                                        dropout = 0.5,
                                        nonlinearity=nn.Tanh
                                    )
        self.decoder = MultVAE_decoder(
                                        item_dim = item_dim,
                                        hidden_dim = hidden_dim, 
                                        latent_dim = latent_dim,
                                        n_hidden_layers = n_dec_hidden_layers,
                                        nonlinearity=nn.Tanh
                                    )
        self.softmax = nn.Softmax(dim = 2)
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        enc_mu, enc_logvar = self.encoder(x)
        z = self.reparameterize(enc_mu, enc_logvar)
        items = self.decoder(z)
        items = self.softmax(items)
        total_int = torch.count_nonzero(x, dim = 2).unsqueeze(dim = 1)
        items =items * total_int.expand(-1,-1, self.item_dim)
        return items,enc_mu, enc_logvar

def VAE_loss_function(x_hat, x, observed, mu, logvar, beta):
    bce = f.binary_cross_entropy(x_hat * observed, x, reduction='sum')
    kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    return bce + beta * kl_div, bce, kl_div
