from lib import *
import gym
import matplotlib.pyplot as plt


test = 0
textlabel = ['SARSA','Q']

if not test:
    env = gym.make('WorldGrid-v0')
    aSARSA = QlearningAlgo(env,.1,.96)
    
    num_episodes = 100
    time_horizon = 100
    for i_episode in range(num_episodes):
        
        obs = env.reset()
        act = aSARSA.greedy(obs)
        
        if i_episode == num_episodes-1:
            aSARSA.activeSave()
        for t in range(time_horizon):
            env.render('human')
            obs_prev = obs
            act_prev = act
            obs, r, done, info = env.step(act)
            act = aSARSA.greedy(obs)
            aSARSA.updateSARSA(act_prev,act,obs_prev,obs,r)
            aSARSA.saveHistory(obs_prev,r,t)
        

        (posX,posY,cr,time) = aSARSA.getHistory()

test = 1
if test:
    env = gym.make('WorldGrid-v0')
    aQ = QlearningAlgo(env,.1,.96)
    
    num_episodes = 100
    time_horizon = 100
    for i_episode in range(num_episodes):
        
        obs = env.reset()
        act = aQ.greedy(obs)
        
        if i_episode == num_episodes-1:
            aQ.activeSave()
        for t in range(time_horizon):
            env.render('human')
            obs_prev = obs
            act_prev = act
            obs, r, done, info = env.step(act)
            act = aQ.greedy(obs)
            aQ.updateQ(act_prev,obs_prev,obs,r)
            aQ.saveHistory(obs_prev,r,t)
        

        (QposX,QposY,Qcr,Qtime) = aQ.getHistory()



##PostProcess
plt.plot(time, cr, 'b')
plt.plot(time, Qcr, 'r')

plt.ylabel(textlabel[0]+' and '+textlabel[1])
plt.xlabel('time')
plt.show()

fig = plt.subplot()
fig.scatter(posX, posY)
plt.plot(posX, posY, 'r--')

plt.ylabel('y')
plt.xlabel('x')
plt.show()

fig = plt.subplot()
fig.scatter(QposX, QposY)
plt.plot(QposX, QposY, 'r--')

plt.ylabel('y')
plt.xlabel('x')
plt.show()
