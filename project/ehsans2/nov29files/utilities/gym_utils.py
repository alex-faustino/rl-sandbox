import gym
import numpy as np

def get_angle(self, cosine, sine):
    possine = 2*(np.array(sine) > 0) - 1
    theta = possine * np.arccos(cosine)
    theta = np.mod(theta, 2*np.pi)
    return theta

def get_space_shape(space):
    if isinstance(space, gym.spaces.discrete.Discrete):
        return [space.n]
    if isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
        return list(space.nvec)
    if isinstance(space, gym.spaces.box.Box):
        return list(space.low.shape)