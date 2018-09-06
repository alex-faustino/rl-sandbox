
from gym.envs.registration import register

register(
    id='GameOfLifeEnv-v0',
    entry_point='GameOFLifeLib.GameOfLife:GOLEnv',
)
