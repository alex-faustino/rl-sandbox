from gym.envs.registration import register

register(
    id='grid_world-v0',
    entry_point='gym_envs.grid_world:Grid',
)

register(
    id='acrobot-v0',
    entry_point='gym_envs.Acrobot_Implementation:AcrobotEnv',
)