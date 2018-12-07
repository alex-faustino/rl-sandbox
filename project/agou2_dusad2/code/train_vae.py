
import numpy as np
import importlib
import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import gym
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torchsummary import summary
from vaelin import VAELin
from vae import VAE
from dataset import VAEDataset
from torch.utils.data import DataLoader, random_split

from PIL import Image

from constants import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = VAE(conv_sizes=[4,8,16,32], device=device).to(device)
model = VAELin(z_size=16, device=device).to(device)

dataset = VAEDataset(transform=Compose([
        ToTensor()
    ]))

train_data, test_data = random_split(dataset, [9000, 1000])

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
train_iter = len(train_data)//batch_size


losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    with tqdm(total=train_iter) as bar:
        for batch_idx, train_batch in enumerate(train_loader):
            train_batch = train_batch.to(device)
            optimizer.zero_grad()
            original, decoded, encoded, z, mu, logvar = model.forward(train_batch)
            
            loss, _, _ = model.loss(original, decoded, mu, logvar)
            losses.append(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            bar.update(1)
            bar.set_postfix(avg_loss=sum(losses[-(batch_idx+1):])/(batch_idx+1))

torch.save(model.state_dict(), VAE_PATH)

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, VAE_PATH)