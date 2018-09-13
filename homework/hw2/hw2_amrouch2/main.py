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
        sumR = 0
        if i_episode == num_episodes-1:
            aSARSA.activeSave()
        for t in range(time_horizon):
            if i_episode == num_episodes-1:
                env.render('human')
            obs_prev = obs
            act_prev = act
            obs, r, done, info = env.step(act)
            act = aSARSA.greedy(obs)
            aSARSA.updateSARSA(act_prev,act,obs_prev,obs,r)
            sumR += r
            aSARSA.saveHistory(obs_prev,sumR,t)
        

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
        sumR = 0
        if i_episode == num_episodes-1:
            aQ.activeSave()
        for t in range(time_horizon):
            if i_episode == num_episodes-1:
                env.render('human')
            obs_prev = obs
            act_prev = act
            obs, r, done, info = env.step(act)
            act = aQ.greedy(obs)
            aQ.updateQ(act_prev,obs_prev,obs,r)
            sumR += r
            aQ.saveHistory(obs_prev,sumR,t)
        

        (QposX,QposY,Qcr,Qtime) = aQ.getHistory()



##PostProcess
plt.plot(time, cr, 'b')
plt.plot(time, Qcr, 'r')

plt.ylabel(textlabel[0]+' and '+textlabel[1]+ ' Cummulative reward')
plt.xlabel('time')
plt.show()

fig = plt.subplot()
fig.scatter(posX, posY)
plt.plot(posX, posY, 'r--')
fig.annotate('start', xy=(posX[0]+0.05, posY[0]+0.05), xytext=(posX[0]+0.5, posY[0]+0.5), arrowprops=dict(facecolor='magenta', shrink=0.01),)
fig.annotate('end', xy=(posX[-1]+0.05, posY[-1]+0.05), xytext=(posX[-1]+0.5, posY[-1]+0.5), arrowprops=dict(facecolor='magenta', shrink=0.01),)
plt.ylabel('y')
plt.xlabel('x')
plt.show()

fig = plt.subplot()
fig.scatter(QposX, QposY)
plt.plot(QposX, QposY, 'r--')
fig.annotate('start', xy=(QposX[0]+0.05, QposY[0]+0.05), xytext=(QposX[0]+0.5, QposY[0]+0.5), arrowprops=dict(facecolor='magenta', shrink=0.01),)
fig.annotate('end', xy=(QposX[-1]+0.05, QposY[-1]+0.05), xytext=(QposX[-1]+0.5, QposY[-1]+0.5), arrowprops=dict(facecolor='magenta', shrink=0.01),)
plt.ylabel('y')
plt.xlabel('x')
plt.show()
