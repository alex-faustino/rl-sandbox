from gym.envs.registration import register

register(
    id='WorldGrid-v0',
    entry_point='GridWorldLib.GridWorld:WORLDGRIDENV',
)
