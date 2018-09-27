from gym.envs.registration import register

register(
    id='MyAcrobot-v0',
    entry_point='my_acrobot.my_acrobot:MyAcrobot',
)
