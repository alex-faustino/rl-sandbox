# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 02:13:52 2018

@author: Vedant
"""

from gym.envs.registration import register, registry, make, spec


# Algorithmic
# ----------------------------------------

register(
    id='attitudeDynamics-v0',
    entry_point='attitudeDynamics.attitudeDynamics:attitudeDynamics',
    
)