import random
import numpy as np

class Qlearning(object):

    def __init__(self, env):
        self.env = env
        self.reset()
        
    def reset(self):
        self.qfunc = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def compute_action(self, cur_pos): 
        if random.random() < self.epsilon: 
            return self.env.action_space.sample() # Explore action space
        
        return np.argmax(self.qfunc[cur_pos]) # Exploit learned values env.action_space.sample() 
        
    def train(self, params): 
        print('Version: ', self.env.env.version)
        
        self.alpha = params['alpha'] ## learning rate
        self.eps_len = params['eps_len'] ## length of each episode
        self.no_eps = params['no_eps'] ## number of episodes/trajectories considered
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.decay = params['decay']
        
        ### start the training algorithm based on Qlearning process
        total_rewards = []
        self.store_state = {}
        self.store_action = {}
        self.store_reward = {}
        #self.store_qfunc = {}
        
        for i_episode in range(self.no_eps): ### 1. Loop for each episode:
            if(i_episode%10==0):
                print('i_eps: ', i_episode)
            
            self.epsilon = self.epsilon*self.decay
            ## Intialize the arrays for storing state, action and reward
            self.store_state[str(i_episode)] = []
            self.store_action[str(i_episode)] = []
            self.store_reward[str(i_episode)] = []
            #self.store_qfunc[str(i_episode)] = []

            ### Initialize the episode with random state
            init_state, cur_pos = self.env.reset() ### 2. Initialize S
            self.store_state[str(i_episode)].append(cur_pos)
            
            #print('Reset state: ', init_state, self.epsilon)
            
            for t in range(self.eps_len): ### 3. Loop for each step of episode:
                action = self.compute_action(cur_pos) ### 4. Choose A from S using policy derived from Q
                self.store_action[str(i_episode)].append(action)
                
                cur_qfunc = self.qfunc[cur_pos, action] ## Q(S,A)        
        
                ### Based on the action, compute the next state and reward obtained
                next_state, reward, done, info = self.env.step(action) ### 5. Take action A, observe R, S'
                self.store_reward[str(i_episode)].append(reward)
        
                next_pos = info['pos'] ## S'
                next_qfunc = np.max(self.qfunc[next_pos]) ## A' = argmax_a Q(S',a)
        
                ### 6. Q(S,A) = Q(S,A) + alpha( R + gamma*Q(S',A')-Q(S,A) )
                q_val = (1 - self.alpha) * cur_qfunc + self.alpha * (reward + self.gamma * next_qfunc)
                self.qfunc[cur_pos, action] = q_val ## Q (S,A)
        
                cur_pos = next_pos ## 7. S<--S'
                self.store_state[str(i_episode)].append(cur_pos)
                
                self.env.render()
                
                if done: 
                    print('Breaking at: ', 'eps_no: ', i_episode, 'eps_len: ', t, 'epsilon: ', self.epsilon)
                    break

            #self.store_qfunc[str(i_episode)] = self.qfunc
            total_rewards.append( sum(self.store_reward[str(i_episode)])/float(t+1) ) #float(self.eps_len) )

        self.env.close()
        return self.qfunc, total_rewards

    def test(self, eps_len):  
        ### Trajectory of the robot in the GridWorld using learned policy via Q learning

        ## Store the position and action taken
        record_pos = []
        record_action = []

        ### Initialize the robot state
        init_state, cur_pos = self.env.reset()
        print('Reset state: ', init_state)

        ## Chose the trained policy and episode len to try the policy on
        for t in range(eps_len):
            record_pos.append(cur_pos)
    
            ### Take action based on the trained policy via SARSA learning
            action = np.argmax(self.qfunc[cur_pos])
            record_action.append(action)
    
            ### Environment is propagated based on action taken
            next_state, reward, done, info = self.env.step(action)
    
            cur_pos = info['pos']    
            self.env.render()
        self.env.close()
        
        return record_pos, record_action
    