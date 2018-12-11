from gym.envs.registration import register

register(
    id='CartPole-v2',
    entry_point='gym_CartpoleEnv.envs:MyCartPoleEnv',
    max_episode_steps=200,
)