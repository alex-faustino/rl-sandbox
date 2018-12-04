#%%
import numpy as np
import gym
import random
from cv2 import resize
import os
from constants import ROLLOUT_DIR, NUM_EPISODES, NUM_STEPS

if not os.path.exists(ROLLOUT_DIR):
    os.makedirs(ROLLOUT_DIR)


env = gym.make('MountainCar-v0')

for ep in range(NUM_EPISODES):
    random_int = random.randint(0, 2**31-1)
    frames = []
    actions = []
    state = env.reset()
    for step in range(NUM_STEPS):
        im = env.render(mode='rgb_array')
        im = resize(im, (HEIGHT, WIDTH))
        frames.append(im)

        action = env.action_space.sample()
        actions.append(action)

        env.step(action)
    frames = np.array(frames, dtype=int)
    actions = np.array(actions, dtype=float)
    filename = os.path.join(ROLLOUT_DIR, str(random_int)+".npz")
    np.savez_compressed(filename, frames=frames, actions=actions)

env.close()


