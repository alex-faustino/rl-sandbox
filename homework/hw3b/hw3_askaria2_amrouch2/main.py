from acrobot import AcrobotEnv
from time import sleep
from NN import DQNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


env = AcrobotEnv()
learn = DQNet()

episode_num = 1
time_horizon = 1000

for i_episode in range(episode_num):
    observation = env.reset()
    init = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    tf.Session().run(init)
    tf.Session().run(init_l)
    for t in range(time_horizon):
        env.render()

        # using greedy algorithm to select action
        action, target_Q = learn.ActSel(observation,1)

        observation, reward, term, x = env.step(action)

        action_old = action
        
        # action selection without greedy
        action, Q_val = learn.ActSel(observation, 0)

        target_Q[action_old] = reward + 0.95*np.max(Q_val)
        
        learn.BackProp(target_Q)
        
    env.close()

