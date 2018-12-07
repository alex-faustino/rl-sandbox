#%%
import torch
import importlib
import dataset
dataset = importlib.reload(dataset)
RNNDataset = dataset.RNNDataset
MultiToTensor = dataset.MultiToTensor
from model.vaelin import VAELin
from constants import *
from torchvision.transforms import ToTensor, Compose
import os
import numpy as np
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
vae = VAELin(z_size=16, device=device).to(device)

loaded = torch.load(VAE_PATH)
vae.load_state_dict(loaded['model_state_dict'])
#%%
dataset = RNNDataset(transform=MultiToTensor())
batch_size = 8
loader = DataLoader(dataset, batch_size=batch_size)

#%%
def to_latent(batch_frames, vae):
    frames = batch_frames.reshape((batch_size, RNN_SAMPLE_SIZE, 3, HEIGHT, WIDTH))
    latent_seq = vae(franes)[3]
    latent_seqs = latent_seq.reshape((batch_size, RNN_SAMPLE_SIZE, LATENT_SIZE))
    return latent_seqs

#%%
for batch in loader:
    frame_seqs = batch['frame_seq'] # (BATCHSIZE, SEQ_LEN, 3, 64, 64)
    action_seqs = batch['action_seq'] # (BATCHSIZE, SEQ_LEN)
    latent_seqs = to_latent(frame_seqs, vae) # (BATCHSIZE, SEQ_LEN, LATENT_SIZE)
    break

#%%
