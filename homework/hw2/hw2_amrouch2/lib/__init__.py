

import numpy as np
from learningAlgoLib import QlearningAlgo

from gym.envs.registration import register

register(
    id='WorldGrid-v0',
    entry_point='lib.GridWorld:WORLDGRIDENV',
)
