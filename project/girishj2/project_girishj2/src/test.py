import gym
import gym_PendulumEnv
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import pickle

class sourcePolicy():
    def __init__(self, sess, state_dim, action_dim, action_bound, params):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='inputs')
        # Recreeate the source Networks
        l1 = tf.nn.relu(tf.matmul(self.inputs, params[0]) + params[1])
        self.mu = self.action_bound* tf.nn.tanh(tf.matmul(l1, params[2]) + params[3])
        self.log_std = tf.nn.softplus(tf.matmul(l1, params[4]) + params[4])

        self.action = self.mu + tf.random_normal(tf.shape(self.mu)) * tf.exp(self.log_std)
    
    def get_action(self, state, stochastic=False):
        if stochastic:
            action =  self.sess.run(self.action, feed_dict={self.inputs: state})[0]
        else:
            action = self.sess.run(self.mu, feed_dict={self.inputs: state})[0]

        return action

env = gym.make('Pendulum-v2')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

file_name = 'PolicyTransfer/pend_NoTransfer_Results'

env.env.set_property(20,0.5,1.1, True)

with tf.Session() as sess:
    
    file_name = 'PPO_Pendulum/pendParams'
    with open(file_name, 'rb') as file:
        net_params = pickle.load(file)

    print(net_params)

    sourceNet = sourcePolicy(sess, state_dim, action_dim, action_bound, net_params)

    sess.run(tf.global_variables_initializer())
    for epoch in range(5):
        running_r = 0.0
        for _ in range(10):
            s = env.reset()        
            for _ in range(200):
                a = sourceNet.get_action(np.reshape(s,(1,state_dim)), False)
                s1,r,done,_ = env.step(a)
                running_r += r
                s = s1
                if done:
                    break
        print(running_r/10)
        Avgreward.append(running_r/10)

    with open(file_name, 'wb') as file:
        pickle.dump(Avgreward, file)

    

