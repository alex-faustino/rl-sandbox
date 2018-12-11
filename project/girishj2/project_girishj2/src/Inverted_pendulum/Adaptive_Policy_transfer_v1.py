import tensorflow as tf
import numpy as np
from replay_buffer import ReplayBuffer  
import gym
import gym_PendulumEnv
import pickle 
import copy

# Define the Training Parameters
MAX_EPOCH = 5000  # Epochs
MAX_EP_LEN = 200  # Steps per Epoch
mini_batch_size = 32 #mini batch data for training
MIN_BUFFER_SIZE = 100 #Min Buffer size
UPDATE_FREQ = 10  # Update freq for trainig network
TRAIN_ITR = 4  # Trainining Steps
BUFFER_SIZE = 10000 # Total Buffer size
TRAIN_STEPS = 4 # Not used
GAMMA = 0.9 # Discount factor
tau = 0.1 # mixing Coeff 
actor_Lr = 1e-3 # Actor Lr
critic_Lr = 1e-2 # Not used
clip_val = 0.2 #policy prob ratio clip value
EPS = 1e-8 #Tikhonov regualrization term

# File name for Saving the results
file_name = 'PolicyTransfer/pendTransfer_Results_4'

# Function to normalize the state
def normalize_state(state_lim, state):
    return state/state_lim

# Source optimal Policy
class sourcePolicy():
    def __init__(self, sess, state_dim, action_dim, action_bound, params):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='inputs')

        # Recreate the source Networks
        ls1 = tf.nn.relu(tf.matmul(self.inputs, params[0]) + params[1])
        self.mu = self.action_bound*tf.nn.tanh(tf.matmul(ls1, params[2]) + params[3])
        self.log_std = tf.nn.softplus(tf.matmul(ls1, params[4]) + params[5])

        self.action = self.mu + tf.random_normal(tf.shape(self.mu)) * tf.exp(self.log_std)

# Total Policy Source + Adaptive Policy    
class TotalPolicy():
    def __init__(self, sess, env, state_dim, action_dim, action_bound, source_Params, adap_params):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.env = env

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='inputs')

        self.source_params = source_Params

        self.adap_params = adap_params

        self.source_mu, self.source_log_std = self.source_net()

        self.adap_mu, self.adap_log_std = self.adaptive_policy()

        self.mu = self.source_mu + self.adap_mu

        self.var = tf.sqrt((tf.exp(self.source_log_std))**2 + (tf.exp(self.adap_log_std))**2)

        self.action = self.mu + tf.random_normal(tf.shape(self.mu)) * self.var

        self.exp_mu = self.env.action_space.sample()

        self.exp_action = (self.exp_mu+self.adap_mu) + tf.random_normal(tf.shape(self.mu))*tf.exp(self.adap_log_std)
    
    def source_net(self):

        params = self.source_params
        # Recreeate the source Networks
        l_t1 = tf.nn.relu(tf.matmul(self.inputs, params[0]) + params[1])
        mu = self.action_bound*tf.nn.tanh(tf.matmul(l_t1, params[2]) + params[3])
        log_std = tf.nn.softplus(tf.matmul(l_t1, params[4]) + params[5])

        return mu, log_std

    def adaptive_policy(self):

        params = self.adap_params
        # Recreeate the source Networks
        l_a1 = tf.nn.relu(tf.matmul(self.inputs, params[0]) + params[1])
        mu = self.action_bound*tf.nn.tanh(tf.matmul(l_a1, params[2]) + params[3])
        log_std = tf.nn.softplus(tf.matmul(l_a1, params[4]) + params[5])
        
        return mu, log_std

    def update_params(self, correction_Params):
        self.adap_params = correction_Params
        
    def get_action(self, state,stochastic=False):
        if stochastic:
            action =  self.sess.run(self.action, feed_dict={self.inputs: state})[0]
        else:
            action = self.sess.run(self.mu, feed_dict={self.inputs: state})[0]

        return action

    def explore_actions(self, state):
        #Sample action
        if np.random.uniform(0.,1.) < 0.0:
            true_mu = self.env.action_space.sample()
            adap_mu, adap_logstd = self.sess.run([self.adap_mu, self.adap_log_std], feed_dict={self.inputs: state})
            action = true_mu + (adap_mu + np.random.normal()*np.exp(adap_logstd))
            delta_action = action - true_mu
            
        else:
            true_mu, true_logstd = self.sess.run([self.source_mu, self.source_log_std], feed_dict={self.inputs: state})
            adap_mu, adap_logstd = self.sess.run([self.adap_mu, self.adap_log_std], feed_dict={self.inputs: state})

            action = true_mu + (adap_mu + np.random.normal()*np.exp(adap_logstd))
            delta_action = action - true_mu

        return action, true_mu, delta_action

# Critic Network Not used
class critic(object):
    def __init__(self, sess, state_dim):

        self.sess  = sess
        self.s_dim = state_dim

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='critic_state')

        l1_critic = tf.layers.dense(self.inputs, 64, tf.nn.relu, name='layer1_critic')

        self.value = tf.layers.dense(l1_critic, 1, name='Value_layer')

        self.discounted_r = tf.placeholder(dtype=tf.float32, shape=[None,1], name='discounted_r')

        self.critic_loss = tf.reduce_mean(tf.square(self.discounted_r-self.value)) 

        self.critic_optimize = tf.train.AdamOptimizer(critic_Lr).minimize(self.critic_loss)

    def get_value(self,state):
        return self.sess.run(self.value, feed_dict={self.inputs:state})

    def train(self, state, discounted_r):
        [self.sess.run(self.critic_optimize, feed_dict={self.inputs:state, self.discounted_r: discounted_r}) for _ in range(TRAIN_STEPS)]

# Aaptive Policy network
class policy(object):
    def __init__(self, sess, policy_name, state_dim, action_dim, action_bound, train_status):

        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.s_dim], name='actor_state')

        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim], name='actions')

        self.mu, self.log_std, self.network_param = self.policy_net(policy_name, train_status)

        self.sigma = tf.exp(self.log_std)

        self.sample_action = self.mu + tf.random_normal(tf.shape(self.mu)) * self.sigma

        self.action_prob_op = self.gaussian_likelihood(self.actions)
        
    def policy_net(self, name ,train_status):

        with tf.variable_scope(name): 
            l1 = tf.layers.dense(self.inputs, 64, tf.nn.relu, trainable=train_status)
            mu = self.action_bound*tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=train_status, name='mu_'+name)
            log_std = tf.layers.dense(l1,self.a_dim, tf.nn.softplus ,trainable=train_status,name ='sigma_'+name )
                
        network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        return mu, log_std, network_params

    def gaussian_likelihood(self, x):
        pre_sum = -0.5 * (((x-self.mu)/(tf.exp(self.log_std)+EPS))**2 + 2*self.log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis = 1)

    def get_action_prob(self, action, state):
        prob = self.sess.run(self.action_prob_op, feed_dict={self.inputs:state, self.actions:action})
        return prob

# Adaptive Meta Learning for reference trajectory tracking
class Ref_Traking_Agent():
    def __init__(self, sess, state_dim, action_dim, action_bound):

        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound

        self.pi= policy(self.sess, 'TT', self.s_dim, self.a_dim, self.action_bound, train_status=True)
        self.old_pi = policy(sess, 'oldp' , self.s_dim, self.a_dim, self.action_bound, train_status=False)

        self.critic_agent = critic(self.sess, self.s_dim)

        self.param_assign_op = [param_old.assign(param_new) for param_old, param_new in zip(self.old_pi.network_param, self.pi.network_param)]

        self.old_action_prob = tf.placeholder(dtype=tf.float32, shape=[None,], name='Old_Action_probs')

        self.prob_ratio = tf.exp(self.pi.action_prob_op - self.old_action_prob)

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        self.tracking_error = tf.placeholder(dtype=tf.float32, shape=[None,], name='error')

        clipped_prob_ratio = tf.clip_by_value(self.prob_ratio, 1.0-clip_val, 1.0+clip_val)

        self.loss = -tf.reduce_mean(tf.minimum(self.prob_ratio, clipped_prob_ratio)*self.tracking_error)

        self.optimize = tf.train.AdamOptimizer(actor_Lr).minimize(self.loss)
    
    def policy_train(self, state, action,old_action_prob,tracking_error):
        [self.sess.run(self.optimize, feed_dict={self.pi.inputs: state, self.pi.actions:action, self.old_action_prob:old_action_prob,self.tracking_error: tracking_error}) for _ in range(TRAIN_ITR)]

    def add_to_buffer(self, state_t, action_t, error_t, done_t, next_state_t):
        self.replay_buffer.add(np.reshape(state_t,(self.s_dim,)),np.reshape(action_t,(self.a_dim,)),error_t, done_t, np.reshape(next_state_t,(self.s_dim,)))

    def get_params(self):
        params = self.sess.run(self.pi.network_param)
        return params

    def update(self, state, action, tracking_error):
        old_action_prob = self.old_pi.get_action_prob(action, state)
        self.sess.run(self.param_assign_op)
        self.policy_train(state,action,old_action_prob, tracking_error)
               
    def get_RAES(self, state, discounted_r):
        gaes = np.vstack(discounted_r) - self.critic_agent.get_value(state)
        gaes = (gaes-np.mean(gaes))/np.var(gaes)
        gaes = np.reshape(gaes, (len(gaes),))
        return gaes
    
    def get_value_target(self, reward_batch, s_next):
        return np.vstack(reward_batch) + GAMMA*np.vstack(self.critic_agent.get_value(s_next))

# Reset the tf graph to default to ensure old data       
tf.reset_default_graph()

def main():
    env_target = gym.make('Pendulum-v2') # Target MDP
    env_source = gym.make('Pendulum-v2') # Source MDP
    env_target.env.set_property(20.,0.5,1.1, True) # Set the properties
    env_source.env.set_property(10.,1.,1.)  # Set the default source properties 
    env_target.seed(0)
    env_source.seed(0)

    # State and Action Dimension
    state_dim = env_target.observation_space.shape[0]
    action_dim = env_target.action_space.shape[0]
    action_bound = env_target.action_space.high[0]

    state_lim =  env_target.observation_space.high

    with tf.Session() as sess:

        agent = Ref_Traking_Agent(sess, state_dim, action_dim, action_bound)

        # Load the saved Source Policy Weights
        source_file_name = 'PPO_Pendulum/pendParams'
        with open(source_file_name, 'rb') as file:
            net_params = pickle.load(file)

        sess.run(tf.global_variables_initializer())

        adaptive_params = agent.get_params()

        targetNet = TotalPolicy(sess, env_target, state_dim, action_dim, action_bound, net_params, adaptive_params)
        
        AvgReward = []

        # Start the Simualation and Training
        for epoch in range(MAX_EPOCH):

            # Initialize target Model and source model
            s_t = env_target.reset()
            hidden_state = env_target.env.get_theta()
            s_s = env_source.reset()
            env_source.env.init_state(hidden_state)
            
            for step in range(MAX_EP_LEN):
                
                # Simulate the Target Env using exploration policy
                action_t, action_s, delta_action = targetNet.explore_actions(np.reshape(s_t, (1,state_dim)))
                s_t1, r_t, done_t, _ = env_target.step(action_t)

                # Simulate Source at target state but optimal policy
                env_source.env.init_state(hidden_state)
                s_s1, r_s, done_s, _ = env_source.step(action_s)

                ns_s1 = normalize_state(state_lim, s_s1)
                ns_t1 = normalize_state(state_lim, s_t1)
            
                r_kl = 100*np.exp(-100*(np.linalg.norm(ns_s1 - ns_t1))**2)
                
                agent.add_to_buffer(s_t, delta_action, r_kl, done_t, s_t1)

                s_t = s_t1

                hidden_state = env_target.env.get_theta()

                if agent.replay_buffer.size() > MIN_BUFFER_SIZE:

                    if (step+1) % UPDATE_FREQ == 0 or step == (MAX_EP_LEN-1):

                        s_batch, a_batch, rkl_batch, t_batch, s1_batch = agent.replay_buffer.sample_batch(mini_batch_size)

                        rkl_batch = (rkl_batch-np.mean(rkl_batch))/(np.var(rkl_batch) +0.001)

                        agent.update(s_batch, a_batch, rkl_batch)

                        targetNet.update_params(agent.get_params())
                        
            # Update the target Network weights before we test
            targetNet.update_params(agent.get_params())
            if epoch % 100 == 0 and epoch > 0:
                running_r = 0.0
                for _ in range(10):
                    s = env_target.reset()
                    for test_step in range(200):
                        action_t = targetNet.get_action(np.reshape(s, (1, state_dim)), stochastic=False)
                        s1,r,done,_ = env_target.step(action_t)
                        running_r += r
                        s = s1
                        if done:
                            break
                AvgReward.append(running_r/10)    
                print('Epoch: %i' %epoch, 'Avg Reward: %i'%AvgReward[-1])

    # Save the result to File for plots
    with open(file_name, 'wb') as file:
        pickle.dump(AvgReward, file)
            
if __name__ == '__main__':
    main()