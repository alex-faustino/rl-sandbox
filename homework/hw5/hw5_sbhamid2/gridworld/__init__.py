from gym.envs.registration import register

register(
    id='MyGridworld-v2',
    entry_point='gridworld.gridworld:GridWorldEnvNew',
    max_episode_steps=500,
    reward_threshold=475.0,
)