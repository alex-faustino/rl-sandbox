import gym
from gym import error, spaces, utils
from gym.utils import seeding

class GridWorld(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    print('GridWorld loaded')

  def step(self, action):
    print('A')

  def reset(self):
    print('A')

  def render(self, mode='human', close=False):
    print('A')
