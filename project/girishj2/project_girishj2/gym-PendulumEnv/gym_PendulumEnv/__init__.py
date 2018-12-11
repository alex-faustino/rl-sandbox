from gym.envs.registration import register

register(
    id='Pendulum-v2',
    entry_point='gym_PendulumEnv.envs:MyPendulumEnv',
    max_episode_steps=200,
)