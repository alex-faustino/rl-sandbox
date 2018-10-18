# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:03:38 2018

@author: Vedant
"""

import time, pickle, os
import numpy as np
import random

def SARSA(env,initial_epsilon = 1, final_epsilon = 0.01,total_episodes = 5000, annealing_period = None,max_steps = 25,lr_rate = 0.99,gamma = 0.9, decay_rate = None):
    if (annealing_period == None):
        annealing_period = total_episodes;
    if(annealing_period > total_episodes):
        print('Annealing period cannot be more than training period, setting annealing period equal to training period')
        annealing_period = total_episodes;
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = initial_epsilon;
    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        return action

    def learn(state, state2, reward, action, action2):
        predict = Q[state, action]
        target = reward + gamma * Q[state2, action2]
        Q[state, action] = predict + lr_rate * (target - predict)
    # Start
    erewards=[]
    for episode in range(total_episodes):
        crewards=0
        t = 0
        state = env.reset()
        action = choose_action(state)
        while t < max_steps:
            #env.render('human')
            state2, reward, done, info = env.step(action)
            action2 = choose_action(state2)
            learn(state, state2, reward, action, action2)
            state = state2
            action = action2
            t += 1
            crewards+=reward
            if done:
                print(episode)
                break
        if decay_rate != None:
            epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * np.exp(-decay_rate * episode) 
        epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * (episode+1)/total_episodes
        erewards.append(crewards);
        
      # os.system('clear')
            #time.sleep(0.1)
    #print ("Score over time: ", rewards/total_episodes)
    #print ("Score in last trial: ", erewards)
    erewards = np.array(erewards)
    return (Q, erewards)
    #with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
        #pickle.dump(Q, f)
        
def QLearn(env,initial_epsilon = 1, final_epsilon = 0.01,total_episodes = 5000, annealing_period = None,max_steps = 25,lr_rate = 0.99,gamma = 0.9, decay_rate = None):
    if (annealing_period == None):
        annealing_period = total_episodes;
    if(annealing_period > total_episodes):
        print('Annealing period cannot be more than training period, setting annealing period equal to training period')
        annealing_period = total_episodes;
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = initial_epsilon;
    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        return action

    def learn(state, state2, reward, action):
        predict = Q[state, action]
        target = reward + gamma * Q[state2, np.argmax(Q[state2])]
        Q[state, action] = predict + lr_rate * (target - predict)
    # Start
    erewards=[]
    for episode in range(total_episodes):
        crewards=0
        t = 0
        state = env.reset()
       
        while t < max_steps:
            action = choose_action(state)
            #env.render('human')
            state2, reward, done, info = env.step(action)
            learn(state, state2, reward, action)
            state = state2
            t += 1
            crewards+=reward
            if done:
                print(episode)
                break
        if decay_rate != None:
            epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * np.exp(-decay_rate * episode) 
        epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * (episode+1)/total_episodes
        erewards.append(crewards);
        #print(episode, epsilon)
      # os.system('clear')
            #time.sleep(0.1)
    #print ("Score over time: ", rewards/total_episodes)
    #print ("Score in last trial: ", erewards)
    return (Q, erewards)
    #with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
        #pickle.dump(Q, f)
        
def Reinforce(env,initial_epsilon = 1, final_epsilon = 0.01,total_episodes =1000, 
              annealing_period = None,max_steps = 25,lr_rate = 0.99, gamma = 1,
              batch_size = 10,decay_rate = None, Imp_samp = False , Causility = False ,
              Base_shift = False):
    
    if (annealing_period == None):
        annealing_period = total_episodes;
    if(annealing_period > total_episodes):
        print('Annealing period cannot be more than training period, setting annealing period equal to training period')
        annealing_period = total_episodes;
    #theta_list = np.ones((env.observation_space.n, env.action_space.n))
    #norm = np.linalg.norm(theta_list,2,0)
    #theta_list = theta_list/norm[0]
    
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    epsilon = initial_epsilon;
    P = np.ones((env.observation_space.n, env.action_space.n))
    
    def apply_softmax(P):
        for q in range(env.observation_space.n):
            P[q] = softmax(P[q])
        return P
    
    P = apply_softmax(P)
    
    def choose_action(state):
        action=0
        
        action = np.random.choice(env.action_space.n , 1 , p = softmax(P[state]))
        action = action[0]
        '''
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.random.choice(env.action_space.n , 1 , p = softmax(P[state]))
            action = action[0]
        '''
        return action

    def update(states, actions, rewards):
        update = np.zeros((env.observation_space.n, env.action_space.n))
        b = np.zeros(len(rewards));
        G=  sum(rewards);
        for i in range(len(rewards)):
            action = np.zeros((env.action_space.n))
            action[actions[i]] = 1.0 
            pmf = P[states[i]]
            grad_ln_pi = action - softmax(pmf)
            if(Causility):
                G =  sum(rewards[i:]);
            if(Base_shift):
                b[t] = sum(rewards[i:]);
            update[states[i]] += grad_ln_pi *G
        return update,b

            #theta_list[states[i]] = theta_list[states[i]]/np.linalg.norm(theta_list[states[i]])
    # Start
    erewards=[]
    action_list = []
    state_list = []
    reward_list = []
    for episode in range(int(total_episodes)):
        
        
        state = env.reset()
        state_list.append(state)
        cumilative_update = np.zeros((env.observation_space.n, env.action_space.n))
        for b in range(batch_size):
            crewards=0.0
            t = 0
            while t < max_steps:
                action = choose_action(state)
                #env.render('human')
    
                state2, reward, done, info = env.step(action)
                state = state2
                action_list.append(action)
                state_list.append(state)
                reward_list.append(reward)
                t += 1
                crewards+=reward
                if done:
                    print(episode)
                    break
            if(Imp_samp):
                states = random.sample(state_list, max_steps)
                actions = random.sample(action_list, max_steps)
                rewards = random.sample(reward_list, max_steps)
            else:
                states = state_list[-max_steps:]
                actions = action_list[-max_steps:]
                rewards = reward_list[-max_steps:]   
            update_val, baseline_val = update(states, actions, rewards)
            cumilative_update += update_val
        P = P + ((1-lr_rate)*(cumilative_update/batch_size))
        #apply_softmax(P)
        #p_from_theta(P,theta_list)
        if decay_rate != None:
            epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * np.exp(-decay_rate * episode) 
        epsilon = initial_epsilon + (final_epsilon - initial_epsilon) * (episode+1)/total_episodes
        erewards.append(crewards/max_steps);
        #print(episode, epsilon)
      # os.system('clear')
            #time.sleep(0.1)
    #print ("Score over time: ", rewards/total_episodes)
    #print ("Score in last trial: ", erewards)
    return (P, erewards)
    #with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
        #pickle.dump(Q, f)