# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 19:08:27 2018
This code is REINFORCE monte carlo algorithm from Sutton and Barto 13.3 example. 
@author: Abdul Alkurdi
"""
import numpy as np

#This is the REINFORCE montecarlo policy-gradient episodic algorithm
def REINFORCE(env, meta_data, difficulty=False):
    ep_len=meta_data[0] #change to 100
    gamma=meta_data[1]
    learning_rate=meta_data[2]
    num_epis_max=meta_data[3] #change to 10,000
    num_actions=meta_data[4]
    num_state=meta_data[5]
    theta=meta_data[6]
    G=[]
    reward_accum=[]
    theta_accum=[]
    
    
    for i in range(num_epis_max): # a loop over episodes for number of episodes desired

        #initialize episode 
        reward=[]
        state=[]
        action=[]
        traj=[]
        sum_grad=np.zeros_like(theta)
        current_state=env.reset()
        action_size=env.action_space.n
        policy=[]
        policy_storage=[]
        current_glp=[] #gradient of the log of probability at each time step
        
        
        for t in range(ep_len): #generating trajectory[i] taso from i=0 to T=100 

            [x,y]=current_state
            s_t=x*5+y # state transform: onehot encoding state by transforming from xy coord to positions in a list
            state.append(s_t) #keeping track of all states
            
            '''#calculating the policy (set of possible actions with probability distribution from given policy parameters
            in this case we are using softmax function to generate the policy from the parameters.'''
            policy=softmax_policy(theta[:,s_t]) 
            policy_storage.append(policy)
            #next we want to find action by sampling from the probability distribution
            current_action=np.random.choice(action_size, p=policy)  #A[i] following pi(.|.,theta)
            action.append(current_action)         
            next_state, step_reward, _ , _ = env.step(current_action,difficulty)
            reward.append(step_reward)
            current_glp=grad_log_softmax(policy,current_action, num_actions)
            sum_grad[:,s_t] += current_glp 
            current_state=next_state
            
        traj.append(zip(state,action))        
        #calculate G, the expectation of future rewards
        G=np.zeros_like(reward)
        G[-1]=reward[-1]
        for i in range(ep_len-2,-1,-1):
            G[i]=G[i+1]*gamma+reward[i]
        #Theta update block
        theta += learning_rate * sum_grad * G[0]
        reward_accum.append(G[0])
        theta_accum.append(theta)
        
        
    return traj, reward_accum, theta_accum, action, state, policy_storage, 'traj, reward_accum, theta_accum, action, state, description string' 

def softmax_policy(theta): 
    '''This function takes in theta and uses softmax to return probability distribution of the policy'''
    for i in theta:
        exp=np.exp(theta) # where theta bar = theta abar, sbar
        policy=exp/np.sum(exp)
    return policy

def grad_log_softmax(policy, action,num_actions):
    softmax_grad=[]
    daa=np.eye(num_actions)
    softmax_grad = (daa[:,action] -policy[:]) 
    # not sure if indexing works this way. it might not slice in the second dimension. you ask to slise all that returns all. check if it is correct.
    # also policy should be size of 25x4, daa is size of 5x5
    return softmax_grad