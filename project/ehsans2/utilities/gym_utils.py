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

def make_env(env_maker_fn, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = env_maker_fn()
        env.seed(seed + rank)
        return env
    from stable_baselines.common import set_global_seeds
    set_global_seeds(seed)
    return _init
