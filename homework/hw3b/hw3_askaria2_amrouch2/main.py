from acrobot import AcroBotMEnv as AcrobotEnv
from time import sleep
from NN import DQNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Memory_Replay_Wrapper(object):
    def __init__(self,max_len=1e4):
        self.scenes = []
        self.max_len = max_len


    def push_scene(self,s_0,a_0,r_0,s_1):
        wrap = {'s0':s_0,
                'a0':a_0,
                'r0':r_0,
                's1':s_1}

        self.scenes.append(wrap)
        if len(self.scenes)> self.max_len:
            rem self.scenes[0]

    def pop_scene(self,idx):
        if idx>self.max_len || idx<0:
            raise ValueError('Wrong Index in scenes')

        scene = self.scenes[idx]

        s0 = scene['s0']
        a0 = scene['a0']    
        r0 = scene['r0']
        s1 = scene['s1']

        return s0,a0,r0,s1

    
if __name__ == '__main__':

    env = AcrobotEnv()
    learn = DQNet()

    episode_num = 1
    time_horizon = 1000
    init = tf.global_variables_initializer()
    sess = learn.Session()
    sess.run(init)

    for i_episode in range(episode_num):
        observation = env.reset()

        for t in range(time_horizon):
            env.render()
            
            # using greedy algorithm to select action
            new_action = learn.ActionSelection(observation, env.action_space.sample, 1 )
            
            observation, reward = env.step(new_action)
            
            action_old = action
            
            # action selection without greedy
            action, Q_val = learn.ActSel(observation, 0, sess.run)
            if t+1 == time_horizon:
                target_Q[action_old] = reward
            else:
                target_Q[action_old] = reward + 0.95*np.max(Q_val)
            
            sess.run(learn.BackProp(target_Q))
            
            env.close()

