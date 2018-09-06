from gym.envs.registration import register

register(
    id='MyAcrobot-v2',
    entry_point='acrobot.acrobot:AcroEnvNew',
    max_episode_steps=500,
    reward_threshold=475.0,
)