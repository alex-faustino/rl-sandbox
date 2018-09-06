import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('acrobot-v0')
env = env.unwrapped
observation = env.reset()
time = 200
actionlist = np.zeros(time)
theta1list = np.zeros(time)
theta2list = np.zeros(time)
dtheta1list = np.zeros(time)
dtheta2list = np.zeros(time)
rewardlist = np.zeros(time)
t = np.zeros(time)
for i in range(time):
    env.render()
    action=env.action_space.sample() # take a random action
    actionlist[i] = action
    state, reward, endding, info = env.step(action)
    theta1list[i] = state[0]
    theta2list[i] = state[1]
    dtheta1list[i] = state[2]
    dtheta2list[i] = state[3]
    rewardlist[i] = reward
    print('state = {}, reward = {}, terminal = {}, info = {}'.format(state, reward, endding, info))
    t[i] = i

plt.plot(t, rewardlist, 'b')
plt.ylabel('reward')
plt.xlabel('time')
plt.show()

plt.plot(t, actionlist, 'b')
plt.ylabel('torque')
plt.xlabel('time')
plt.show()

plt.plot(t, theta1list, 'b')
plt.plot(t, theta2list, 'r')
plt.ylabel('angle')
plt.xlabel('time')
plt.show()

plt.plot(t, dtheta1list, 'b')
plt.plot(t, dtheta2list, 'r')
plt.ylabel('angular velocity')
plt.xlabel('time')
plt.show()
