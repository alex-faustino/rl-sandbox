#%%
import numpy as np
import torch
from torch.utils.data import Dataset
from imageio import imread
import os
from constants import DATASET_SIZE, HEIGHT, WIDTH, ROLLOUT_DIR

class VAEDataset(Dataset):
    def __init__(self, rollout_dir=ROLLOUT_DIR, size=DATASET_SIZE, transform=None):
        self.transform=transform
        self.frames = np.zeros((size, HEIGHT, WIDTH, 3), dtype=np.uint8)
        episodes = [ep for ep in os.listdir(ROLLOUT_DIR) if '.npz' in ep]
        num_frames = 0
        for episode in episodes:
            ep_frames = np.load(os.path.join(ROLLOUT_DIR, episode))['frames']
            self.frames[num_frames:num_frames + len(ep_frames)] = ep_frames
            num_frames = min(num_frames + len(ep_frames), size)
            if num_frames == size:
                break

    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        sample = self.frames[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

