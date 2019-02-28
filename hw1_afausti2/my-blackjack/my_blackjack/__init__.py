from gym.envs.registration import register

register(
    id='MyBlackjack-v0',
    entry_point='my_blackjack.envs:MyBlackjackEnv',
)
