from gym.envs.registration import register

register(
    id='MyLocalizeRobot-v2',
    entry_point='localizerobot.localizerobot:LocalizeEnvNew',
    max_episode_steps=500,
    reward_threshold=475.0,
)