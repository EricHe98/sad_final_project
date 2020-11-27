import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as numpy

class MultVAE_encoder(nn.module):
	def __init(self, item_dim, latent_dim, n_hidden_layers, dropout = 0.5):
		self.item_dim = item_dim
		self.latent_dim = latent_dim
		self.first_layer = nn.Linear(in_features = item_dim, out_features = 600)

class MultVAE_decoder(nn.module):
	def __init(self, item_dim, latent_dim, n_hidden_layers, dropout = 0.5):
		self.item_dim = item_dim
		self.latent_dim = latent_dim
		self.first_layer = nn.Linear(in_features = item_dim, out_features = 600)


class MultVae(nn.module):
	def __init__(self, item_dim, latent_dim, n_hidden_layers,)




def VAE_loss_function(x_hat, x, mu, logvar, beta = 1.00):
	bce = f.binary_cross_entropy(x_hat, x)
	kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
	return bce + beta * kl_div