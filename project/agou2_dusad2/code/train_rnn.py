#%%
import os
import numpy as np
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader

from dataset import RNNDataset, MultiToTensor
from model.vaelin import VAELin
from model.mdnrnn import MDNRNN
from constants import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = RNNDataset(transform=MultiToTensor())
loader = DataLoader(dataset, batch_size=RNN_BATCH_SIZE, drop_last=True, num_workers=4)
vae = VAELin(z_size=LATENT_SIZE, device=device).to(device)
loaded = torch.load(VAE_PATH, map_location=device)
vae.load_state_dict(loaded['model_state_dict'])
#%%

model = MDNRNN().to(device)
#%%
def to_latent(batch_frames, vae):
    frames = batch_frames.reshape((RNN_BATCH_SIZE, RNN_SEQ_LEN, 3, HEIGHT, WIDTH))
    latent_seq = vae(frames)[3]
    latent_seqs = latent_seq.reshape((RNN_BATCH_SIZE, RNN_SEQ_LEN, LATENT_SIZE))
    return latent_seqs

losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#%%
for epoch in range(RNN_EPOCHS):
    with tqdm(enumerate(loader), total=len(loader)) as progress:
        old_rnn_hidden = None
        for batch_idx, batch in progress:
            optimizer.zero_grad()
            frame_seqs = batch['frame_seq'].to(device).float() # (BATCHSIZE, SEQ_LEN, 3, 64, 64)
            action_seqs = batch['action_seq'].to(device).float().unsqueeze(2) # (BATCHSIZE, SEQ_LEN)
            latent_seqs = to_latent(frame_seqs, vae) # (BATCHSIZE, SEQ_LEN, LATENT_SIZE)

            curr_latent = latent_seqs[:,:-1]
            curr_actions = action_seqs[:,:-1]
            next_latent = latent_seqs[:,1:]
            mus, logsigmas, logpis, rnn_out, rnn_hidden = model(curr_latent, curr_actions, old_rnn_hidden)

            loss = model.loss(next_latent, mus, logsigmas, logpis)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            optimizer.step()
            
            old_rnn_hidden = (torch.tensor(rnn_hidden[0]).detach(), torch.tensor(rnn_hidden[1]).detach())
            
            progress.set_postfix(epoch=epoch, avg_loss=sum(losses[-(batch_idx+1):])/(batch_idx+1))
print(losses[-1])
#%%
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, RNN_PATH)
