import random
import numpy as np

class REINFORCElearning(object):

    def __init__(self, env):
        self.env = env
        self.reset()
                
    def reset(self):
        self.pfunc = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_p = []
        self.batch_idx = []
        self.idx = 0

    def compute_likelihood(self, state, pfunc):
        pmf_smax = np.exp(pfunc[state,:]) / np.sum(np.exp(pfunc[state,:]), axis=0)
        return pmf_smax 
    
    def compute_action(self, state, pfunc): 
        ### policy based action gradient
        pmf_smax = self.compute_likelihood(state, pfunc) #np.exp(pfunc[state,:]) / np.sum(np.exp(pfunc[state,:]), axis=0)
        a = np.random.choice(np.arange(pfunc.shape[1]), 1, replace=False, p=pmf_smax)
        action =  int(a)
        prob = pmf_smax[a]
        
        return action, prob
    
    def compute_gradient(self, state, action, pfunc):
        dJ_grad = np.zeros(pfunc.shape) 
        ### softmax layer 
        pmf_smax = self.compute_likelihood(state, pfunc) #np.exp(pfunc[state,:]) / np.sum(np.exp(pfunc[state,:]), axis=0)
        ### Finding Delta_grad-> \delta_{ss} and \delta_{aa}
        dJ_grad[state,:] = -pmf_smax[action]
        dJ_grad[state, action] = 1 - pmf_smax[action]
        return dJ_grad

    def compute_episode(self, pfunc):
        
        store_s = np.zeros([self.eps_len], dtype=int)
        store_a = np.zeros([self.eps_len], dtype=int)
        store_r = np.zeros([self.eps_len], dtype=int)   
        store_p = np.zeros([self.eps_len])   
        
        init_state, cur_pos = self.env.reset()
        store_s[0] = cur_pos
        
        for t in range(self.eps_len): 
            ### policy based action gradient
            store_a[t], store_p[t] = self.compute_action(cur_pos, pfunc)
        
            ### Environment is propagated based on action taken
            next_state, store_r[t], done, info = self.env.step(store_a[t])
            next_pos = info['pos'] ## S'            
            if (t<self.eps_len-1):
                store_s[t+1] = next_pos
            
            cur_pos = next_pos ## S<--S
            self.env.render()
        
        return store_s, store_a, store_r, store_p
    
    def compute_batch(self, no_traj, pfunc): 
        
        for n in range(no_traj):
            store_s, store_a, store_r, store_p = self.compute_episode(pfunc)
            
            if(len(self.batch_s) >= self.mem_size): 
                self.batch_s.pop(0)
                self.batch_a.pop(0)
                self.batch_r.pop(0)
                self.batch_p.pop(0)
                self.batch_idx.pop(0)
                
            self.batch_s.append(store_s)
            self.batch_a.append(store_a)
            self.batch_r.append(store_r)
            self.batch_p.append(store_p)
            self.batch_idx.append([self.idx]*len(store_s))
            #print('no_traj: ', n, no_traj, len(self.batch_s))
            
        return self.batch_s, self.batch_a, self.batch_r, self.batch_p, self.batch_idx
    
    def compute_baseline(self):         
        store_b = np.zeros([self.eps_len])
        for t in range(self.eps_len):
            for k in range(self.mem_size):
                store_b[t] += np.sum(self.batch_r[k][t:])/float(self.mem_size)
        
        return store_b
    
    def train(self, params): 
        print('Version: ', self.env.env.version)

        self.no_eps = params['no_eps'] ## number of episodes/trajectories considered
        self.batch_size = params['batch_size']
        self.alpha = params['alpha'] ## learning rate
        self.eps_len = params['eps_len'] ## length of each episode
        self.algo = params['algo'] ## what algorithm options: baseline-b, importance sampling-i, causality-c
        self.mem_size = params['mem_size'] ## baseline length
        
        if (self.algo.find('b')!=-1): 
            print('Executing Baseline')
        if (self.algo.find('c')!=-1): 
            print('Executing Causality')
        if (self.algo.find('i')!=-1): 
            print('Executing Importance Sampling')

        ### start the training algorithm based on Qlearning process
        total_rewards = []
        no_batch = int(self.no_eps/self.batch_size)

        ### Initialize the algorithm by filling the buffer with trajectories based on the memory length
        batch_s, batch_a, batch_r, batch_p, batch_idx = self.compute_batch(self.mem_size, self.pfunc)
        
        batch_pfunc = {}
        batch_dJgrad = {}
        for i_batch in range(no_batch): 
            ### Update the index of batch number and also the corresponding pfunc used at this instant 
            self.idx = i_batch
            print('Batch: ', self.idx)
            
            ### set the policy used for all the episodes in the batch
            batch_pfunc[str(i_batch)] = self.pfunc

            dJ_grad_batch = np.zeros(self.pfunc.shape) ### Gradient has same shape as policy func        
            rewards_batch = 0 ### initialize rewards_batch and dJ_grad_batch

            batch_s, batch_a, batch_r, batch_p, batch_idx = self.compute_batch(self.batch_size, batch_pfunc[str(i_batch)] )
            
            #print('Batch index: ', batch_idx)
            
            ### Baseline estimation 
            if (self.algo.find('b')!=-1): 
                #print('Baseline')
                store_b = self.compute_baseline()
            else: 
                store_b = np.zeros([self.eps_len])
            
            ### here!!! work from here
            if (self.algo.find('i')!=-1): 
                #print('Importance Sampling')
                store_i = random.sample([i for i in range(self.mem_size)], self.batch_size)
                #np.random.choice(self.mem_size, self.batch_size)
            else: 
                store_i = np.arange(self.mem_size-self.batch_size, self.mem_size)
            
            store_s = np.zeros([self.eps_len], dtype=int)
            store_a = np.zeros([self.eps_len], dtype=int)
            store_r = np.zeros([self.eps_len], dtype=int)   
            store_p = np.zeros([self.eps_len])  
            store_idx = np.zeros([self.eps_len], dtype=int)   
        
            ### In each batch, going over the trajctories
            for n in range(self.batch_size):
                ## Getting one trajectory 
                store_s = batch_s[store_i[n]]
                store_a = batch_a[store_i[n]]
                store_r = batch_r[store_i[n]]
                store_p = batch_p[store_i[n]]
                store_idx = batch_idx[store_i[n]]
                
                dJ_grad_eps = np.zeros(batch_pfunc[str(i_batch)].shape) ### initialize Dj_grad_eps for each episode
                rewards_eps = np.sum(store_r)
                lratio_den = 1                
                for t in range(self.eps_len):                     
                    ### Executing causality and baseline shifts
                    if (self.algo.find('c')!=-1): 
                        rewards_c = np.sum(store_r[t:]) - store_b[t]
                    else: 
                        rewards_c = np.sum(store_r) - store_b[t]
                    
                    #print('Values: ', t, store_idx)
                    dJ_grad_eps += self.compute_gradient(store_s[t], store_a[t],  batch_pfunc[str(store_idx[t])])*rewards_c
                    pmf_eps = self.compute_likelihood(store_s[t], batch_pfunc[str(i_batch)])
                    lratio_den *= pmf_eps[store_a[t]]
                                
                lratio_num = np.prod(store_p)
                lratio = lratio_num/lratio_den
                #print('lratio: ', lratio)
                if (lratio>100): 
                    continue
                dJ_grad_batch += (lratio*dJ_grad_eps)/float(self.batch_size)
                rewards_batch += (rewards_eps)/float(self.batch_size)
        
            #update policy
            total_rewards.append(rewards_batch/float(self.eps_len) )
            batch_dJgrad[str(i_batch)] = dJ_grad_batch
            self.pfunc = self.pfunc + self.alpha * dJ_grad_batch
            
        self.env.close()
        return self.pfunc, total_rewards, batch_dJgrad        
        
    def train_org(self, params): 
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
