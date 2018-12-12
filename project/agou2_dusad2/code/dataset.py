#%%
import numpy as np
import torch
from torch.utils.data import Dataset
from imageio import imread
import os
from constants import *
from torchvision.transforms import ToTensor

class VAEDataset(Dataset):
    def __init__(self, rollout_dir=ROLLOUT_DIR, size=VAE_FULL_DATASET_SIZE, transform=None):
        self.transform=transform
        self.frames = np.zeros((size, HEIGHT, WIDTH, 3), dtype=np.uint8)
        episodes = [ep for ep in os.listdir(ROLLOUT_DIR) if '.npz' in ep]
        num_frames = 0
        for episode in episodes:
            ep_frames = np.load(os.path.join(ROLLOUT_DIR, episode))['frames']
            start, end = num_frames, min(num_frames + len(ep_frames), len(self.frames))
            self.frames[num_frames:num_frames + len(ep_frames)] = ep_frames[:end-start]
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

class RNNDataset(Dataset):
    def __init__(self, rollout_dir=ROLLOUT_DIR, size=RNN_DATASET_SIZE, sample_size=RNN_SEQ_LEN, transform=None):
        self.transform = transform
        self.sample_size = sample_size
        episodes = [ep for ep in os.listdir(ROLLOUT_DIR) if '.npz' in ep]
        self.frame_seqs = []
        self.action_seqs = []
        num_seq = 0
        for episode in episodes:
            
            saved = np.load(os.path.join(ROLLOUT_DIR, episode))
            frames = saved['frames']
            if len(frames) < sample_size:
                continue
            num_seq += 1
            self.frame_seqs.append(frames)
            actions = saved['actions']
            self.action_seqs.append(actions)
        self.frame_seqs = np.array(self.frame_seqs)
        self.action_seqs = np.array(self.action_seqs)

    def __len__(self):
        return len(self.frame_seqs)
    
    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, len(self.frame_seqs[idx]) - self.sample_size)
        sample = {
            'frame_seq': self.frame_seqs[idx, sample_idx:sample_idx + self.sample_size],
            'action_seq': self.action_seqs[idx, sample_idx:sample_idx + self.sample_size]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

class MultiToTensor:
    '''
    Converts a batch of numpy images to tensors
    '''

    def __call__(self, sample):
        frame_seq, action_seq = sample['frame_seq'], sample['action_seq']
        frame_seq = torch.from_numpy(frame_seq).permute((0,3,1,2)).float().div(255)
        return {'frame_seq': frame_seq, 'action_seq': action_seq}
