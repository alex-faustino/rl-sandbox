'''
# ------------------------------------------
AE 598RL Homework-3
Author: Girish Joshi
Email: girishj2@illinois.edu
This Code implements DQN on CartPole Environment
#-------------------------------------------
'''
import gym
import numpy as np
import tensorflow as tf
import pickle
from replay_buffer import ReplayBuffer
import random
import gc
gc.enable()

#DQN Learning Paramters
MAX_EPOCH = 5000
MAX_LENGTH = 200
net_lr = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 32

# Results Storing File Name
file_name = 'Results_DQN_test3'

# Network Paramaters Number of hidden units
n_hidden_layer_1 = 100
n_hidden_layer_2 = 100

def weight_init(shape, var_name):
        initial_val = tf.truncated_normal(shape)
        return tf.Variable(initial_val, name=var_name)

def bias_init(shape, var_name):
    initial_val = tf.constant(0.01,shape=shape)
    return tf.Variable(initial_val, name=var_name)

class DQN(object):

    def __init__(self,sess,learning_rate, buffer_size):
        self.sess = sess
        self.gamma = 0.9
        self.epsilon_min = 0.01
        self.epsilon_max = 0.5
        self.epsilon_decay = 0.999
        self.epsilon = self.epsilon_max
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.tau = 0.01

        # Define Environment
        self.env = gym.make('CartPole-v0')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        
        self.inputs, self.out = self.createDQN(primary_net=True)

        self.network_params = tf.trainable_variables()

        # Create Target Network
        self.target_inputs, self.target_out = self.createDQN(primary_net=False)

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Update Target Network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)\
            + tf.multiply(self.target_network_params[i], 1. - self.tau)) for i in range(len(self.target_network_params))]

        self.predicted_q = tf.placeholder(tf.float32,[None])
        self.action_inputs = tf.placeholder(tf.float32,[None,self.action_dim])
        self.Q_action = tf.reduce_sum(tf.multiply(self.out,self.action_inputs), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.predicted_q-self.Q_action))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        
    def createDQN(self, primary_net):

        inputs = tf.placeholder(tf.float32,[None,self.state_dim], name='inputs')
        
        # Input layer --> Hidden Layer 1
        w1 = weight_init([self.state_dim,n_hidden_layer_1],'w1')
        b1 = bias_init([n_hidden_layer_1],'b1')

        # Hidden layer 1 --> Hidden Layer 2
        w2 = weight_init([n_hidden_layer_1,n_hidden_layer_2],'w2')
        b2 = bias_init([n_hidden_layer_2],'b2')

        # Hidden Layer --> output
        w3 = weight_init([n_hidden_layer_2,self.action_dim],'w3')
        b3 = bias_init([self.action_dim],'b3')

        # Create the 1st layer of Neural Network
        h1 = tf.nn.tanh(tf.matmul(inputs,w1) + b1) 

        # Create the 2nd Layer of Neural Nework (Action Inserted in the second layer)
        h2 = tf.nn.tanh(tf.matmul(h1,w2) + b2)

        # Create the output layer of the Neural Network
        out = tf.add(tf.matmul(h2,w3), b3, name='Net_output')

        if primary_net:
            #Save File
            self.save_file = "DQN_Results/DQN_net"
            #Network Save
            self.saver = tf.train.Saver()
            self.saver = tf.train.Saver([w1,w2,w3,b1,b2,b3])

        return inputs, out

    def add_to_buffer(self, state, action, reward, done, next_state):
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1
        self.replay_buffer.add(np.reshape(state,(self.state_dim,)),np.reshape(action_one_hot,(self.action_dim,)),reward,done,np.reshape(next_state,(self.state_dim,)))

    def trainDQN(self,mini_batch_size):
        if self.replay_buffer.size() >= mini_batch_size:
            s_batch,a_batch,r_batch,t_batch,s2_batch =  self.replay_buffer.sample_batch(mini_batch_size) 
            target_q = self.predict_target(s2_batch)
            y_i = []

            for k in range(mini_batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.gamma*np.max(target_q[k]))
            self.sess.run(self.optimize, feed_dict={self.inputs:s_batch, self.action_inputs: a_batch, self.predicted_q: np.reshape(y_i, (BATCH_SIZE))})
            self.sess.run(self.update_target_network_params)

    def egreedy_action(self,s):
        Q_value = self.sess.run(self.out, feed_dict={self.inputs:np.reshape(s,[1,self.state_dim])})
        if np.random.uniform(0,1,size=None) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(Q_value[0])

    def predict(self,s):
        return self.sess.run(self.out, feed_dict={self.inputs:np.reshape(s,[BATCH_SIZE,self.state_dim])})

    def predict_target(self,s):
        return self.sess.run(self.target_out, feed_dict={self.target_inputs:np.reshape(s,[BATCH_SIZE,self.state_dim])})

    def get_action(self,s):
        Q_value = self.sess.run(self.out, feed_dict={self.inputs:np.reshape(s,[BATCH_SIZE,self.state_dim])})
        return np.argmax(Q_value[0])

def main():
    with tf.Session() as sess:
        
        agent = DQN(sess,net_lr,BUFFER_SIZE)
        
        sess.run(tf.global_variables_initializer())

        # Reward vector for Ploting
        epReward = []
        avgReward = []
        
        for epochs in range(MAX_EPOCH):
            s = agent.env.reset()
            total_reward = 0
            for step in range(MAX_LENGTH):
                action = agent.egreedy_action(s)
                s2,r,done,_ = agent.env.step(action)
                agent.add_to_buffer(s,action,r,done,s2)
                agent.trainDQN(BATCH_SIZE)
                total_reward += r 
                s = s2
                if done:
                    break
            agent.epsilon *= agent.epsilon_decay
            if agent.epsilon <= agent.epsilon_min:
                agent.epsilon = agent.epsilon_min
            epReward.append(total_reward)
    
            if epochs % 100 == 0 and epochs > 0:
                avgReward.append(np.mean(epReward[-100:]))
                print('Epoch:',epochs,'Average Reward:',avgReward[-1],'Epsilon', agent.epsilon)
                

        #Save the Network
        agent.saver.save(sess,agent.save_file)

        with open(file_name, 'wb') as file:
            pickle.dump(avgReward,file)

if __name__ == '__main__':
    main()



   