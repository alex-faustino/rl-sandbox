import gym
env = gym.make('acrobot-v0')
env = env.unwrapped
observation = env.reset()
for i in range(200):
    env.render()
    action=env.action_space.sample() # take a random action
    state, reward, endding, info = env.step(action)
    print('state = {}, reward = {}, terminal = {}, info = {}'.format(state, reward, endding, info))
