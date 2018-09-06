from gym.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point='gym.envs.grid_world:GridWorldEnv',
    max_episode_steps=200,
    reward_threshold=100.0,
)

register(
    id='Drop-v0',
    entry_point='gym.envs.drop:DropEnv',
    max_episode_steps=200,
    reward_threshold=100.0,
)
