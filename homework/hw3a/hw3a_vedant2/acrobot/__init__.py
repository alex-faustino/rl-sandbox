# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:37:17 2018

@author: Vedant
"""

from gym.envs.registration import register, registry, make, spec


# Algorithmic
# ----------------------------------------

register(
    id='vedant_acrobot-v0',
    entry_point='acrobot.acrobot_vedant:acrobot_vedant',
    max_episode_steps=50000,
)