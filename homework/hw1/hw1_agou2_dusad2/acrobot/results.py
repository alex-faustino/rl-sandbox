from acrobot import AcrobotEnv

env = AcrobotEnv()
env.reset()
for _ in range(1000):
    print(env.render())
    env.step(env.action_space.sample()) # take a random action