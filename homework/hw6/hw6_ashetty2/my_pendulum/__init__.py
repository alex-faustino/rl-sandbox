from gym.envs.registration import register

register(
    id='MyPendulum-v0',
    entry_point='my_pendulum.my_pendulum:MyPendulum',
)
