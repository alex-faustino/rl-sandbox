import tensorflow as tf 
import numpy as np
from replay_buffer import ReplayBuffer 
import matplotlib.pyplot as plt
import gym
import pickle
import gc
gc.enable()

MAX_EPOCH = 10000
MAX_EP_LEN = 200
mini_batch_size = 32
BUFFER_SIZE = 10000
TRAIN_STEPS = 4
GAMMA = 0.9
actor_Lr = 0.5e-4
critic_Lr = 1e-3
clip_val = 0.2
c2 = 0.01
kl_target = 0.5

# File Name for saving the Results to file
file_name = 'hw6_PPO_Pendulum_v'

class critic(object):
    def __init__(self, sess, state_dim):

        self.sess  = sess
        self.s_dim = state_dim

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='critic_state')

        l1_critic = tf.layers.dense(self.inputs, 128, tf.nn.relu, name='layer1_critic')

        #l2_critic = tf.layers.dense(l1_critic, 50, tf.nn.relu, name='layer2_critic')

        self.value = tf.layers.dense(l1_critic, 1, name='Value_layer')

        self.discounted_r = tf.placeholder(dtype=tf.float32, shape=[None,1], name='discounted_r')

        self.critic_loss = tf.reduce_mean(tf.square(self.discounted_r-self.value)) 

        self.critic_optimize = tf.train.AdamOptimizer(critic_Lr).minimize(self.critic_loss)

    def get_value(self,state):
        return self.sess.run(self.value, feed_dict={self.inputs:state})

    def train(self, state, discounted_r):
        [self.sess.run(self.critic_optimize, feed_dict={self.inputs:state, self.discounted_r: discounted_r}) for _ in range(TRAIN_STEPS)]


class policy(object):
    def __init__(self, sess, policy_name, state_dim, action_dim, action_bound, train_status):

        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='actor_state')

        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim], name='actions')

        self.actor_prob, self.network_param = self.policy_net(policy_name, train_status)

        self.sample_action = tf.squeeze(self.actor_prob.sample(1), axis=0)

        self.action_prob_op = self.actor_prob.log_prob(self.actions)
        
    def policy_net(self, name ,train_status):

        with tf.variable_scope(name): 
            l1 = tf.layers.dense(self.inputs,300, tf.nn.relu, trainable=train_status)
            #l2 = tf.layers.dense(l1, 64, tf.nn.relu, trainable=train_status)
            mu = 2.0*tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=train_status, name='mu_'+name)
            sigma = tf.layers.dense(l1,self.a_dim, tf.nn.softplus, trainable=train_status,name ='sigma_'+name )
            actor_prob = tf.distributions.Normal(loc=mu, scale=sigma)

        network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return actor_prob, network_params
    
    def get_action(self,state):
        a = self.sess.run(self.sample_action, feed_dict={self.inputs: state})[0]
        return np.clip(a, -self.action_bound, self.action_bound)

    def get_action_prob(self, action, state):
        prob = self.sess.run(self.action_prob_op, feed_dict={self.inputs:state, self.actions:action})
        return prob
    
class PPO:
    def __init__(self, sess, state_dim, action_dim, action_bound):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        
        self.new_policy = policy(sess, 'newp' ,state_dim, action_dim, action_bound, train_status=True)
        self.old_policy = policy(sess, 'oldp' ,state_dim, action_dim, action_bound, train_status=False)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.critic_agent  =  critic(sess, state_dim)

        self.param_assign_op = [param_old.assign(param_new) for param_old, param_new in zip(self.old_policy.network_param, self.new_policy.network_param)]

        self.old_action_prob = tf.placeholder(dtype=tf.float32, shape=[None,1], name='Old_Action_probs')

        #self.prob_ratio = tf.exp(tf.log(tf.clip_by_value(self.new_policy.action_prob_op, 1e-10, 1.)) - tf.log(tf.clip_by_value(self.old_action_prob, 1e-10, 1.)))

        self.prob_ratio = tf.exp(self.new_policy.action_prob_op - self.old_action_prob)

        self.GAE = tf.placeholder(dtype=tf.float32, shape=[None,], name='GAE')

        clipped_prob_ratio = tf.clip_by_value(self.prob_ratio, 1.0-clip_val, 1.0+clip_val)

        loss_clip = -tf.reduce_mean(tf.minimum(self.prob_ratio, clipped_prob_ratio)*self.GAE)
        
        # Entropy Loss function
        #entropy =  tf.reduce_mean(self.new_policy.action_prob_op*tf.log(tf.clip_by_value(self.new_policy.action_prob_op, 1e-10, 1.0)))
        entropy =  tf.reduce_mean(tf.exp(self.new_policy.action_prob_op)*self.new_policy.action_prob_op)

        # KL divergence between new and old policies loss function
        #loss_kl = tf.reduce_mean(tf.exp(self.new_policy.action_prob_op)*(self.old_action_prob-self.new_policy.action_prob_op))
        
        loss = loss_clip + c2*entropy

        self.optimize = tf.train.AdamOptimizer(actor_Lr).minimize(loss)

    def update(self, state, action, discounted_r):
        old_action_prob = self.old_policy.get_action_prob(action, state)
        self.sess.run(self.param_assign_op)
        GAE = self.get_GAES(state, discounted_r)
        self.train(state,action,old_action_prob, GAE)
        self.critic_agent.train(state, discounted_r)

    def train(self, state,action,old_action_prob,GAE):
        [self.sess.run(self.optimize, feed_dict={self.new_policy.inputs:state, self.new_policy.actions:action, self.old_action_prob:old_action_prob, self.GAE: GAE}) for _ in range(TRAIN_STEPS)]

    def discounted_r(self, reward_batch, s_next):
        return np.vstack(reward_batch) + GAMMA*np.vstack(self.critic_agent.get_value(s_next))

    def get_GAES(self, state, discounted_r):
        gaes = np.vstack(discounted_r) - self.critic_agent.get_value(state)
        gaes = np.reshape(gaes, (len(gaes),))
        return gaes

    def add_to_buffer(self, state, action, reward, next_state):
        self.replay_buffer.add(np.reshape(state,(self.s_dim,)),np.reshape(action,(self.a_dim,)),reward,np.reshape(next_state,(self.s_dim,)))
        

def main():
    env = gym.make('Pendulum-v0')
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    with tf.Session() as sess:
        agent = PPO(sess, state_dim, action_dim, action_bound)
        sess.run(tf.global_variables_initializer())
        Epoch_total_reward = []
        AvgReward = []

        # Start the Simualation and Training
        for epoch in range(MAX_EPOCH):
            s = env.reset()
            running_r = 0.0 
            for step in range(MAX_EP_LEN):
                action = agent.new_policy.get_action(np.reshape(s,(1,state_dim)))
                s1,r,done,_ = env.step(action)
                agent.add_to_buffer(s,action,r,s1)
                running_r += r
                #env.render()
                s = s1
                if (step+1) % mini_batch_size == 0 or step == MAX_EP_LEN-1:

                    s_batch,a_batch,r_batch,s2_batch =  agent.replay_buffer.sample_batch(mini_batch_size) 

                    discounted_r = agent.discounted_r(r_batch, s2_batch)   

                    agent.update(s_batch, a_batch, discounted_r)

            #Append the Total Return on Policy
            Epoch_total_reward.append(running_r)
                    
            if epoch % 100 == 0 and epoch > 0:
                AvgReward.append(np.mean(Epoch_total_reward[-100:]))
                print('Epoch: %i' %epoch, 'Avg Reward: %i'%AvgReward[-1])

    # Save the Average Reward Vector          
    with open(file_name,'wb') as file:
        pickle.dump(AvgReward, file)
    
    #Plot the Average Reward Vector
    plt.plot(AvgReward)
    plt.show()

if __name__ == '__main__':
    main()
