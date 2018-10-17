'''
# ------------------------------------------
AE 598RL Homework-5
Author: Girish Joshi
Email: girishj2@illinois.edu
This Code implements the Importance Sampling Policy Gradient Algorithm on the Grid World Problem

The Grid world Environment is developed as part of Homework-1
Use the File: myGridworld.py
#-------------------------------------------
'''

import tensorflow as tf
import numpy as np
from myGridworld import gridWorldEnv
import matplotlib.pyplot as plt
from collections import deque
import pickle

easy = False

# Initialize the model and learning Parameters
env = gridWorldEnv(easy)
state_dim = 2
action_dim = 4

# Discount factor
GAMMA = 1
# Leanring rate
Lr = 1e-2
#bias
bias = 0.

#Maximum Episodes
MAX_EPISODE = 5000
# Length of Each Episode
LEN_EPISODE = 100

#Sample Trajectories number for update
UPDATE_FREQ = 5

# Network Hidden Layer Size
n_hidden_layer = 25

def weight_variable(shape):
    initial_val = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial_val)

def bias_variable(shape):
    initial_val = tf.constant(0.03,shape=shape)
    return tf.Variable(initial_val)

def discount_rewards(r):
    discounted_reward = np.zeros_like(r)
    running_reward = 0
    for t in reversed(range(0,r.size)):
        running_reward = running_reward*GAMMA + r[t]
        discounted_reward[t] = running_reward
    return discounted_reward

class PolicyGradient():
    
    def __init__(self,state_dim,action_dim,Lr):
        
        # Network Inputs
        self.inputs = tf.placeholder(dtype=tf.float32,shape=[None,state_dim], name='State_inputs')
        # Input Layer
        w1 = weight_variable([state_dim,n_hidden_layer])
        b1 = bias_variable([n_hidden_layer])

        # output layer
        w2 = weight_variable([n_hidden_layer,action_dim])
        b2 = bias_variable([action_dim])

        # 1st Hidden Layer
        h1 = tf.nn.relu(tf.matmul(self.inputs,w1)+b1)
        # Output Layer
        self.out = tf.nn.softmax(tf.matmul(h1,w2)+b2,name='Net_outputs')
        
        
        self.reward_holder = tf.placeholder(dtype=tf.float32,shape=[None])

        self.action_holder = tf.placeholder(dtype=tf.int32,shape=[None])

        self.prob_ratio_holder = tf.placeholder(dtype=tf.float32, shape=[None])
        
        self.indexes = tf.range(0,tf.shape(self.out)[0])*tf.shape(self.out)[1] + self.action_holder
        
        self.responsible_outs = tf.gather(tf.reshape(self.out,[-1]), self.indexes)

        self.loss = -tf.reduce_mean(self.prob_ratio_holder*tf.log(self.responsible_outs)*self.reward_holder)
        
        self.network_params = tf.trainable_variables()
        self.gradient_holder = []

        # Save File
        #self.save_file = "PG_gridwordl/PGgrdiWorld_net"
        # Network Save
        #self.saver = tf.train.Saver()
        
        for idx, grad in enumerate(self.network_params):
            grad_placeholder = tf.placeholder(tf.float32)
            self.gradient_holder.append(grad_placeholder)
            
        self.net_gradient = tf.gradients(self.loss,self.network_params)
        self.optimize = tf.train.AdamOptimizer(Lr).apply_gradients(zip(self.gradient_holder,self.network_params))

class behavioralPolicy():
    def __init__(self, state_dim, action_dim):
        
        self.inputs = tf.placeholder(dtype=tf.float32,shape=[None,state_dim])
        self.w1 = tf.truncated_normal([state_dim, n_hidden_layer], mean=0.0, stddev=0.01)
        self.b1 = tf.truncated_normal([n_hidden_layer],mean=0.,stddev=0.01)
        self.w2 = tf.truncated_normal([n_hidden_layer, action_dim], mean=0.0, stddev=0.01)
        self.b2 = tf.truncated_normal([action_dim], mean=0., stddev=0.01)
        
        # 1st Hidden Layer
        h1 = tf.nn.relu(tf.matmul(self.inputs,self.w1)+self.b1)
        # Output Layer
        self.out = tf.nn.softmax(tf.matmul(h1,self.w2)+self.b2)

    def updateWeights(self,network_params):
        self.w1 = network_params[0]
        self.b1 = network_params[1]
        self.w2 = network_params[2]
        self.b2 = network_params[3]    

tf.reset_default_graph() #Clear the Tensorflow graph.
agent = PolicyGradient(state_dim, action_dim, Lr)
behavioral_agent = behavioralPolicy(state_dim, action_dim)

with tf.Session() as sess:

    #Set Random Seed for repeatability
    np.random.seed(1234)
    tf.set_random_seed(1234)

    sess.run(tf.global_variables_initializer())
    total_reward = []
    total_length = []
    AvgReward = []

    if easy:
        # File Name for saving the Results to file
        file_name = 'hw5_IS_PGgridWorldmlp_easyon'
    else:
        file_name = 'hw5_IS_PGgridWorldmlp_easyoff'

    
    gradBuffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx] = grad*0
    
    for epoch in range(MAX_EPISODE):
        s = env.reset()
        ep_history = deque()
       
        for ep_length in range(LEN_EPISODE):
            
            # Using Behavioral policy
            #action_prob = sess.run(behavioral_agent.out, feed_dict={behavioral_agent.inputs:np.reshape(s,(1,state_dim))})
            #action_prob_selected = np.random.choice(action_prob[0], p=action_prob[0])

            #Using Uniform over all policies
            action_prob = np.random.uniform(low=0.,high=1.0,size=[4,1])
            action_prob_selected = action_prob[0]
            action = np.argmax(action_prob[0])
            
            # Step environment with action selected
            s1,r,done = env.step(action)

            #Calculate the probility for selected action using actual policy for IS ratio
            true_policy_action_prob = sess.run(agent.out, feed_dict={agent.inputs:np.reshape(s,(1,state_dim))})[0]
            prob_ratio = true_policy_action_prob[action]/action_prob_selected[action]

            # Add trajectory to history
            ep_history.append((np.reshape(s,(state_dim,)),action,r-bias,prob_ratio))
            s = s1
            if ep_length == LEN_EPISODE-1:
                s_batch = np.array([_[0] for _ in ep_history])
                a_batch = np.array([_[1] for _ in ep_history])
                r_batch = np.array([_[2] for _ in ep_history])
                prob_ratio_batch = np.array([_[3] for _ in ep_history])
                r_batch = discount_rewards(r_batch)
                
                grads = sess.run(agent.net_gradient,feed_dict={agent.inputs:s_batch,
                                                                agent.reward_holder:r_batch,
                                                                agent.action_holder:a_batch,
                                                                agent.prob_ratio_holder:prob_ratio_batch})
                                                            
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad
                    
                if epoch % UPDATE_FREQ == 0 and epoch!=0:
                    feed_dict = dict(zip(agent.gradient_holder, gradBuffer))
                    sess.run(agent.optimize, feed_dict=feed_dict)
                    for idx, grad in enumerate(gradBuffer):
                        gradBuffer[idx] = grad*0
                
        s = env.reset()
        running_r = 0
        for i in range(LEN_EPISODE):
            action_prob = sess.run(agent.out, feed_dict={agent.inputs:np.reshape(s,(1,state_dim))})
            action_prob_selected = np.random.choice(action_prob[0], p=action_prob[0])
            action = np.argmax(action_prob == action_prob_selected)
            s1,r,done = env.step(action)
            s = s1
            running_r += r
            if i == LEN_EPISODE-1:
                total_reward.append(running_r)
                bias = running_r/LEN_EPISODE
                break

        if epoch % 100 == 0 and epoch > 0:
            AvgReward.append(np.mean(total_reward[-100:]))
            print('Epoch:',epoch, 'Avg Reward:',np.mean(total_reward[-100:]))

    #agent.saver.save(sess,agent.save_file)
    # Save the results to file
    with open(file_name,'wb') as file:
        pickle.dump(AvgReward, file)
