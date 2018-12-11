#%%
from tqdm import tqdm

# numpy
import numpy as np
from PIL import Image

# gym
import gym

# torch stuff
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, ToPILImage
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchsummary import summary
import torch.nn.functional as F

# our stuff
import importlib
from model.mdrnn import MDRNN
from model.vaelin import VAELin

from constants import *

# cma
from cma import CMAEvolutionStrategy as CMAES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("MountainCarContinuous-v0")

# load vae and rnn
vae = VAELin(z_size=LATENT_SIZE, device=device).to(device)
vae.load_state_dict(torch.load(VAE_PATH, map_location=device)['model_state_dict'])
rnn = MDRNN(
    sequence_length=500,
    hidden_space_dim=32,
    action_space_dim=1,
    latent_space_dim=LATENT_SIZE,
    num_mixtures=10,
    rnn_type="lstm",
    n_layers=5
)
rnn.load_state_dict(torch.load(RNN_PATH, map_location=device)['model_state_dict'])

#%%

def to_latent(vae, frame):
    frame = frame.unsqueeze(0)
    latent_seq = vae(frame)[3].detach()
    return latent_seq.squeeze()

def get_action(latent, solution, hidden=None):
    w= solution[:-1]
    b = solution[-1]

    latent = latent.cpu().numpy()
    if hidden is None:
        hidden_size = len(solution) - len(latent) - 1
        hidden = np.zeros(hidden_size)
    else:
        hidden = torch.cat([hidden[0], hidden[1]], -1).float().detach().squeeze(1).view(-1).cpu().numpy()

    stacked_features = np.hstack([latent, hidden])
    res = w @stacked_features  + b
    return np.tanh(res)

def get_loss(solution):
    max_pos = -0.4

    rewards = 0
    for ep in range(10):
        state = env.reset()
        old_rnn_hidden = None
        for step in range(NUM_STEPS):
            im = env.render(mode='rgb_array')
            im = Image.fromarray(im).resize((HEIGHT, WIDTH), Image.BILINEAR)
            frame = ToTensor()(im)
            latent = to_latent(vae, frame)
            action = get_action(latent, solution, old_rnn_hidden)
            _, _, _, _, rnn_hidden = rnn(latent.unsqueeze(0).unsqueeze(0), torch.tensor([[[action]]]).float(), old_rnn_hidden)
            old_rnn_hidden = rnn_hidden
            state, reward, done, _ = env.step([action])
            if state[0] > max_pos:
                max_pos = state[0]
                reward += 10
            rewards += reward
            if done:
                break
    loss = -rewards
    return loss

env = gym.make("MountainCarContinuous-v0")
param_size = LATENT_SIZE + 32*2*5 + 1
init_params = param_size*[0]
init_sigma = 1
popsize = 64
es = CMAES(init_params, init_sigma, {'popsize':popsize})
#%%
es.optimize(get_loss)
env.close()

