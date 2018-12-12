import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal

from constants import *



class GMM(nn.Module):
    def __init__(self, n_mixtures, hidden_size, latent_space_dim):
        super(GMM, self).__init__()
        self.n_mixtures = n_mixtures
        self.net = nn.Linear(hidden_size, (latent_space_dim*2 + 1)*n_mixtures)
        self.stride = latent_space_dim*n_mixtures
        self.latent_space_dim = latent_space_dim
        self.hidden_size = hidden_size

    def loss(self, batch, mus, logsigmas, logpis):
        sigmas = torch.exp(logsigmas)
        batch = batch.unsqueeze(-2)
        dist = Normal(mus, sigmas)
        
        batch_log_probs = dist.log_prob(batch)
        batch_log_probs = logpis + torch.sum(batch_log_probs, dim=-1)
        log_prob = torch.logsumexp(batch_log_probs, dim=-1, keepdim=True)
        return -log_prob.mean()

    def forward(self, input):
        out = self.net(input)
        mus = out[:, :,  :self.stride]
        mus = mus.view(mus.shape[0], mus.shape[1], -1, self.latent_space_dim)

        logsigmas = out[:, :,  self.stride:2*self.stride]
        logsigmas = logsigmas.view(logsigmas.shape[0], logsigmas.shape[1], -1, self.latent_space_dim)

        logpis = out[:, :, -self.n_mixtures:]
        return mus, logsigmas, logpis



class MDNRNN(nn.Module):
    def __init__(self,
        sequence_length=1000, 
        hidden_space_dim=RNN_HIDDEN_SIZE,
        action_space_dim=1, 
        latent_space_dim=LATENT_SIZE,
        num_mixtures=RNN_NUM_MIXTURES, 
        rnn_type="lstm", 
        n_layers=RNN_NUM_LAYERS):

        super(MDNRNN, self).__init__()
        
        self.rnn_type = rnn_type
        
        models = {
            "lstm": nn.LSTM,
            "rnn": nn.RNN,
            "gru": nn.GRU
        }
        
        self.sequence_length = sequence_length
        self.n_layers = n_layers
        self.hidden_space_dim = hidden_space_dim
        
        self.latent_space_dim = latent_space_dim
        self.action_space_dim = action_space_dim
        self.num_mixtures = num_mixtures
        
        
        self.input_shape = latent_space_dim + action_space_dim
        self.output_shape = latent_space_dim


        self.rnn = models[rnn_type](latent_space_dim + action_space_dim, hidden_space_dim, n_layers)
        self.gmm = GMM(num_mixtures, hidden_space_dim, latent_space_dim)
        
    
    def forward(self, latents, actions, prev_rnn_hidden=None):
        inputs = torch.cat([actions, latents], -1)
        rnn_out, rnn_hidden = self.rnn(inputs, prev_rnn_hidden) # hidden_state is zero 
        mus, logsigmas, logpis = self.gmm(rnn_out)
        return mus, logsigmas, logpis, rnn_out, rnn_hidden

    def loss(self, next_states, mus, logsigmas, logpis):
        return self.gmm.loss(next_states, mus, logsigmas, logpis)

def sample_gmm(mus, logsigmas, logpis):
    mu, logsigma, logpi = mus[0,-1], logsigmas[0,-1], logpis[0,-1]
    choice = torch.distributions.categorical.Categorical(F.softmax(logpi, dim=0)).sample()
    dist = torch.distributions.normal.Normal(mu[choice], logsigma[choice].exp())
    return dist.sample()