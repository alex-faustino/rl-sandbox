# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:30:50 2018

@author: Vedant
"""

import gym
import gridWorld

env = gym.make('gridWorld-v0')
MAP = ["EAEBE",
        "ESSSE",
        "ESSSE",
        "ESSSE"]

nA = 4;
nS = 5 * 5;
desc = np.asarray(MAP,dtype='c');
isd = np.array(MAP == b'S').astype('float64').ravel();
isd /= isd.sum();
env = gym.make('FrozenLake8x8-v0')