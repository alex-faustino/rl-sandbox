# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:21:42 2018

@author: Vedant
"""

from gym.envs.registration import register, registry, make, spec


# Algorithmic
# ----------------------------------------

register(
    id='gridWorld-v0',
    entry_point='gridWorld.gridWorld:gridWorld',
)

register(
    id='hard_gridWorld-v0',
    entry_point='gridWorld.gridWorld:gridWorld',
    kwargs={'hard': True},
)