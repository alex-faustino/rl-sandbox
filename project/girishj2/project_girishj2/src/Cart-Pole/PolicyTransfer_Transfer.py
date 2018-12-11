import tensorflow as tf
import numpy as np
import gym
import gym_CartpoleEnv
import matplotlib.pyplot as plt
from collections import deque
import pickle

# Initialize the model and learning Parameters
envt = gym.make('CartPole-v2')
envs = gym.make('CartPole-v2')
envt.env.set_prop(9.8,1.0,0.1,0.75,10)
envs.env.set_prop(9.8,1.0,0.1,0.5,10)
state_dim = envt.observation_space.shape[0]
action_dim = envt.action_space.n

# Discount factor
GAMMA = 0.99
# Leanring rate
Lr = 5e-3
Lr_decay = 0.999

#Epsilon and epsilon decay term
epsilon = 0.9
epsilon_decay = 0.999

#Maximum Episodes
MAX_EPISODE = 5000
# Length of Each Episode
LEN_EPISODE = 999

#Sample Trajectories number for update
UPDATE_FREQ = 1

# Network Hidden Layer Size
n_hidden_layer = 10

# Results Storing File Name
file_name = 'CartPole_Results/cpTransfer_Results4'

def weight_variable(shape):
    initial_val = tf.truncated_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial_val)

def bias_variable(shape):
    initial_val = tf.constant(0.03,shape=shape)
    return tf.Variable(initial_val)

def discount_rewards(r, gamma):
    discounted_reward = np.zeros_like(r)
    running_reward = 0
    for t in reversed(range(0,r.size)):
        running_reward = running_reward*gamma + r[t]
        discounted_reward[t] = running_reward
    return discounted_reward

def reward_generate(s_source, s_target):
    reward = 3.4*np.exp(-100.0*np.mean((s_source-s_target)**2))
    return reward

class SourcePolicy():

    def __init__(self,sess):

        # Load Source Optimal Policy Network
        saver_source = tf.train.import_meta_graph('Source_PolicyPG/Source_PG_net.meta')
        saver_source.restore(sess,tf.train.latest_checkpoint('Source_PolicyPG/'))
        graph1 = tf.get_default_graph()
        self.source_policy = graph1.get_tensor_by_name('Net_outputs:0')
        self.source_inputs = graph1.get_tensor_by_name('State_inputs:0')

class PolicyGradient():
    
    def __init__(self,state_dim,action_dim,Lr):
        
        # Network Inputs
        self.inputs = tf.placeholder(dtype=tf.float32,shape=[None,state_dim], name='PG_inputs')
        # Input Layer
        wp1 = weight_variable([state_dim,n_hidden_layer])
        bp1 = bias_variable([n_hidden_layer])

        # output layer
        wp2 = weight_variable([n_hidden_layer,action_dim])
        bp2 = bias_variable([action_dim])

        # 1st Hidden Layer
        hp1 = tf.nn.relu(tf.matmul(self.inputs,wp1)+bp1)
        # Output Layer
        self.policy_out = tf.nn.softmax(tf.matmul(hp1,wp2)+bp2,name='PG_outputs')
        
        self.reward_holder = tf.placeholder(dtype=tf.float32,shape=[None])
        self.action_holder = tf.placeholder(dtype=tf.float32,shape=[None,action_dim])
        
        self.responsible_outs = tf.reduce_sum(self.policy_out*self.action_holder,1)
        
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outs)*self.reward_holder)
        
        self.network_params = tf.trainable_variables()
        self.gradient_holder = []

        # Save File
        #self.save_file = "PGTransfer/PGT_net"
        # Network Save
        #self.saver = tf.train.Saver()
        
        for idx, grad in enumerate(self.network_params):
            grad_placeholder = tf.placeholder(tf.float32)
            self.gradient_holder.append(grad_placeholder)
            
        self.net_gradient = tf.gradients(self.loss,self.network_params)
        self.optimize = tf.train.AdamOptimizer(Lr).apply_gradients(zip(self.gradient_holder,self.network_params))


tf.reset_default_graph() #Clear the Tensorflow graph.
agent = PolicyGradient(state_dim, action_dim, Lr)

with tf.Session() as sess:

    # Set Random Seed for repeatability
    np.random.seed(1234)
    tf.set_random_seed(1234)
    envt.seed(1234)

    sess.run(tf.global_variables_initializer())
    total_reward = []
    total_reward_target = []
    total_length = []
    Avg_Reward_History = []
    Avg_TargetReward_History = []

    source = SourcePolicy(sess)
    
    gradBuffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx] = grad*0
    
    for epoch in range(MAX_EPISODE):
        st = envt.reset()
        ss = envs.reset()
        envs.env.init_state(st)
        running_r = 0
        running_rt = 0
        Lr *= Lr_decay
        ep_history = deque()
        for ep_length in range(LEN_EPISODE):
            action_hot = np.zeros((1,2))
            action_prob = sess.run(agent.policy_out, feed_dict={agent.inputs:np.reshape(st,(1,state_dim))})
            action_prob_selected = np.random.choice(action_prob[0], p=action_prob[0])
            action_t = np.argmax(action_prob == action_prob_selected)
            action_hot[0][action_t] = 1
            st1,rt,donet,info = envt.step(action_t)
            action_prob = sess.run(source.source_policy, feed_dict={source.source_inputs:np.reshape(st,(1,state_dim))})
            action_s = np.argmax(action_prob)
            envs.env.init_state(st)
            ss1,rs,dones,info = envs.step(action_s)
            r = reward_generate(ss1,st1)
            ep_history.append((np.reshape(st,(state_dim,)),np.reshape(action_hot,(action_dim,)),r,rt))
            st = st1
            running_r += r
            running_rt += rt
            if donet or dones:
                
                s_batch = np.array([_[0] for _ in ep_history])
                a_batch = np.array([_[1] for _ in ep_history])
                r_batch = np.array([_[2] for _ in ep_history])
                rt_batch = np.array([_[3] for _ in ep_history])
                #r_batch = discount_rewards(r_batch,1.1)
                rt_batch = discount_rewards(rt_batch,GAMMA)
                total_r_batch = 0.3*rt_batch + 0.7*r_batch
                
                grads = sess.run(agent.net_gradient,feed_dict={agent.inputs:s_batch,
                                                                agent.reward_holder:total_r_batch,
                                                                agent.action_holder:a_batch})
                #print(grads)                                            
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad
                    
                if epoch % UPDATE_FREQ == 0 and epoch!=0:
                    feed_dict = dict(zip(agent.gradient_holder, gradBuffer))
                    sess.run(agent.optimize, feed_dict=feed_dict)
                    for idx, grad in enumerate(gradBuffer):
                        gradBuffer[idx] = grad*0
                total_reward.append(running_r)
                total_reward_target.append(running_rt)
                total_length.append(ep_length)
                break
        if epoch % 100 == 0:
            Avg_Reward_History.append(np.mean(total_reward[-100:]))
            Avg_TargetReward_History.append(np.mean(total_reward_target[-100:]))
            print('Epoch: %i' %epoch, 'Avg Reward: %i'%np.mean(total_reward_target[-100:]))

    #agent.saver.save(sess,agent.save_file)

    with open(file_name,'wb') as file:
        pickle.dump(Avg_TargetReward_History, file)

    # testing the poicy
    # st = envt.reset()
    # for i in range(500):
    #     envt.render()
    #     action_prob = sess.run(agent.policy_out, feed_dict={agent.inputs:np.reshape(st,(1,state_dim))})
    #     action_t = np.argmax(action_prob)
    #     st1,r,done,info = envt.step(action_t)
    #     st = st1
    #     if done:
    #         break

# plt.figure(1)
# plt.plot(Avg_Reward_History)
# plt.show()
