
import numpy as np
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import gym
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torchsummary import summary
from model.vaelin import VAELin
from model.vae import VAE
from dataset import VAEDataset
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from PIL import Image

from constants import *
from resample_vae import VAEResampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

# load dataset
all_data = VAEDataset(size=VAE_FULL_DATASET_SIZE, transform=ToTensor())

vae_sample_weights = VAEResampler(all_data).get_sample_weights()
sampler = WeightedRandomSampler(vae_sample_weights, num_samples=VAE_USE_DATASET_SIZE, replacement=True)
train_loader = DataLoader(all_data, batch_size=VAE_BATCH_SIZE, sampler=sampler, num_workers=2)

model = VAELin(z_size=LATENT_SIZE, device=device).to(device)
losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = VAE_EPOCHS

for epoch in range(num_epochs):
    with tqdm(enumerate(train_loader), total=len(train_loader)) as progress:
        for batch_idx, train_batch in progress:
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            original, decoded, encoded, z, mu, logvar = model.forward(train_batch)
            
            loss, _, _ = model.loss(original, decoded, mu, logvar)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            progress.set_postfix(avg_loss=sum(losses[-(batch_idx+1):])/(batch_idx+1))

torch.save(model.state_dict(), VAE_PATH)

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, VAE_PATH)