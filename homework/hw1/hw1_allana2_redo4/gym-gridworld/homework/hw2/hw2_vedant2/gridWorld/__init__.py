# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:21:42 2018

@author: Vedant
"""

from gym.envs.registration import register, registry, make, spec


# Algorithmic
# ----------------------------------------

register(
    id='GridWorld-v0',
    entry_point='gridWorld.gridWorld2:gridWorld',
)

register(
    id='hard_GridWorld-v0',
    entry_point='gridWorld.gridWorld2:gridWorld',
    kwargs={'hard': True},
)
