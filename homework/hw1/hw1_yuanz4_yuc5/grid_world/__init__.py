from gym.envs.registration import register

register(
    id='grid-v0',
    entry_point='grid_world.envs:GridEnv',
)
register(
    id='grid-extrahard-v0',
    entry_point='grid_world.envs:GridExtraHardEnv',
)
