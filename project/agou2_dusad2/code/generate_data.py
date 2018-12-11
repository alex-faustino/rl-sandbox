#%%
import numpy as np
import gym
import random
import os
from constants import ROLLOUT_DIR, NUM_EPISODES, NUM_STEPS, HEIGHT, WIDTH
from PIL import Image

from tqdm import tqdm

if not os.path.exists(ROLLOUT_DIR):
    os.makedirs(ROLLOUT_DIR)

env = gym.make('MountainCar-v0')

for ep in tqdm(range(NUM_EPISODES)):
    random_int = random.randint(0, 2**31-1)
    frames = []
    actions = []
    rewards = []
    state = env.reset()
    for step in range(NUM_STEPS):
        im = env.render(mode='rgb_array')
        im = Image.fromarray(im).resize((HEIGHT, WIDTH), Image.BILINEAR)
        frames.append(np.array(im))

        action = env.action_space.sample()
        actions.append(action)

        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    
    rewards = np.array(rewards, dtype=float)
    frames = np.array(frames, dtype=np.uint8)
    actions = np.array(actions, dtype=float)
    
    print(rewards.shape, frames.shape)
    filename = os.path.join(ROLLOUT_DIR, str(random_int)+".npz")
    np.savez_compressed(filename, frames=frames, actions=actions, rewards=rewards)

env.close()



#%%
