from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gym_gridworld.envs:GridWorldEnv',
)

register(
    id='GridWorldHard-v0',
    entry_point='gym_gridworld.envs:GridWorldHardEnv',
)
