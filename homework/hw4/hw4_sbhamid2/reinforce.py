import random
import numpy as np

class REINFORCElearning(object):

    def __init__(self, env):
        self.env = env
        self.reset()
        
    def reset(self):
        self.pfunc = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def compute_action(self, state, pfunc): 
        ### policy based action gradient
        pmf_smax = np.exp(pfunc[state,:]) / np.sum(np.exp(pfunc[state,:]), axis=0)
        a = np.random.choice(np.arange(pfunc.shape[1]), 1, replace=False, p=pmf_smax)
        action =  int(a) 
        return action
    
    def compute_gradient(self, state, action, pfunc):
        dJ_grad = np.zeros(pfunc.shape) 
        ### softmax layer 
        pmf_smax = np.exp(pfunc[state,:]) / np.sum(np.exp(pfunc[state,:]), axis=0)
        ### Finding Delta_grad-> \delta_{ss} and \delta_{aa}
        dJ_grad[state,:] = -pmf_smax[action]
        dJ_grad[state, action] = 1 - pmf_smax[action]
        return dJ_grad

    def compute_episode(self, pfunc):
        
        store_s = np.zeros([self.eps_len], dtype=int)
        store_a = np.zeros([self.eps_len], dtype=int)
        store_r = np.zeros([self.eps_len], dtype=int)   
        
        init_state, cur_pos = self.env.reset()
        store_s[0] = cur_pos
        
        for t in range(self.eps_len):        
            ### policy based action gradient
            store_a[t] = self.compute_action(cur_pos, pfunc)
        
            ### Environment is propagated based on action taken
            next_state, store_r[t], done, info = self.env.step(store_a[t])
            next_pos = info['pos'] ## S'            
            if (t<self.eps_len-1):
                store_s[t+1] = next_pos
            
            cur_pos = next_pos ## S<--S
            self.env.render()
        
        return store_s, store_a, store_r
        
    def train(self, params): 
        print('Version: ', self.env.env.version)
        
        self.no_eps = params['no_eps'] ## number of episodes/trajectories considered
        self.batch_size = params['batch_size']
        self.alpha = params['alpha'] ## learning rate
        self.eps_len = params['eps_len'] ## length of each episode

        ### start the training algorithm based on Qlearning process
        total_rewards = []
        no_batch = int(self.no_eps/self.batch_size)
        
        for i_batch in range(no_batch): 
            pfunc_batch = self.pfunc ### set the policy used for all the episodes in the batch
            dJ_grad_batch = np.zeros(self.pfunc.shape) ### Gradient has same shape as policy func        
            rewards_batch = 0 ### initialize rewards_batch and dJ_grad_batch
            
            for n in range(self.batch_size):
                store_s, store_a, store_r = self.compute_episode(pfunc_batch)
                
                dJ_grad_eps = np.zeros(self.pfunc.shape) ### initialize Dj_grad_eps and rewards_eps for each episode
                rewards_eps = np.sum(store_r)
                for t in range(self.eps_len): 
                    dJ_grad_eps += self.compute_gradient(store_s[t], store_a[t], pfunc_batch)
        
                dJ_grad_batch += (dJ_grad_eps*rewards_eps)/float(self.batch_size)
                rewards_batch += (rewards_eps)/float(self.batch_size)
        
            #update policy
            total_rewards.append(rewards_batch/float(self.eps_len) )
            self.pfunc = self.pfunc + self.alpha * dJ_grad_batch

        self.env.close()
        return self.pfunc, total_rewards
    
    def test(self, eps_len):  
        ### Trajectory of the robot in the GridWorld using learned policy via SARSA learning

        ## Store the position and action taken
        record_pos = []
        record_action = []

        ### Initialize the robot state
        init_state, cur_pos = self.env.reset()
        print('Reset state: ', init_state)

        ## Chose the trained policy and episode len to try the policy on
        for t in range(eps_len):
            record_pos.append(cur_pos)
    
            ### Take action based on the trained policy via REINFORCE learning

            ### policy based action gradient
            pmf_smax = np.exp(self.pfunc[cur_pos,:]) / np.sum(np.exp(self.pfunc[cur_pos,:]), axis=0)
            
            #### Here, trained_policy is the trained policy
            a = np.random.choice(np.arange(self.pfunc.shape[1]), 1, replace=False, p=pmf_smax)
            action = int(a)    
            record_action.append(action)
    
            ### Environment is propagated based on action taken
            next_state, reward, done, info = self.env.step(action)    
            cur_pos = info['pos']    
            
            self.env.render()
            
        self.env.close()
        return record_pos, record_action
