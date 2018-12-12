from gym.envs.registration import register

register(
    id='MyUrbanWorld-v1',
    entry_point='urbanworld.urbanworld:UrbanWorldEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)