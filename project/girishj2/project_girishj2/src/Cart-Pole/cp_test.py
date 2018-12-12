import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import gym_CartpoleEnv
import matplotlib.pyplot as plt
from collections import deque
import pickle

# Initialize the model and learning Parameters
env = gym.make('CartPole-v2')
env.env.set_prop(9.8,1.0,0.1,1.3,10)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

file_name = 'CartPole_Results/cpNoTransfer_Results1'

class SourcePolicy():

    def __init__(self,sess):

        # Load Source Optimal Policy Network
        saver_source = tf.train.import_meta_graph('Source_PolicyPG/Source_PG_net.meta')
        saver_source.restore(sess,tf.train.latest_checkpoint('Source_PolicyPG/'))
        graph1 = tf.get_default_graph()
        self.source_policy = graph1.get_tensor_by_name('Net_outputs:0')
        self.source_inputs = graph1.get_tensor_by_name('State_inputs:0')

    

with tf.Session() as sess:

    source = SourcePolicy(sess)
    AvgReward = []

    for epoch in range(50):

        running_r = 0.

        for episodes in range(10):
            s = env.reset()
            for steps in range(999):
                action_prob = sess.run(source.source_policy, feed_dict={source.source_inputs:np.reshape(s,(1,state_dim))})
                action = np.argmax(action_prob)
                s_1,r,done,info = env.step(action)
                running_r += r
                s = s_1
                if done:
                    break
                
        AvgReward.append(running_r/10)
        print('Epoch %i'%epoch, 'Avg Reward %i'%AvgReward[-1])

    with open(file_name,'wb') as file:
        pickle.dump(AvgReward, file)