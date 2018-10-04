import numpy as np
import matplotlib.pyplot as plt
import pylab
import gym
import sys
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.ticker as plticker

class gridworld(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self): 
    self.render_label = 'f'
    self.gridnum = int(7) #size of gridworld
    self.location_x = np.random.randint(int(1),self.gridnum-1)
    self.location_y = np.random.randint(int(1),self.gridnum-1)
    self.action = int(0) # no initial action until computed
    self.previous_action = self.action
    self.previous_x = self.location_x
    self.previous_y = self.location_y
    self.previous_previous_x = self.previous_x
    self.previous_previous_y = self.previous_y
    self.allowed_actions = np.array([1,2,3,4])[np.newaxis]
    self.actionSet = np.matrix([1,2,3,4])
    self.episode_length = int(100)
    self.num_episodes = int(10000) # currently only running one episode, 10000
    self.my_alpha = 0.1
    self.my_gamma = 0.9
    self.my_epsilon = 0.1
    self.my_reward =np.array([ [-1,-1,-1,-1,-1,-1,-1],[-1, 0, 10, 0, 5, 0, -1],[-1, 0, 0, 0, 0, 0, -1],\
    [-1, 0, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, 0, -1],[-1, 0, 0, 0, 0, 0, -1],[-1, -1, -1, -1, -1, -1, -1] ]) 
    self.my_reward = np.flipud(self.my_reward)
    self.my_reward_model = np.zeros([self.gridnum,self.gridnum])#reward model updated based on observations
    self.my_q_function = np.random.rand(self.gridnum,self.gridnum, int(4))# randomly initialized via reward model
    self.my_q_function[0::6,:,:] = 0# to prevent biasing agent with favorable "out of bound" q functions
    self.my_q_function[:,0::6,:] = 0# to prevent biasing agent with favorable "out of bound" q functions
    self.my_reward_log = np.random.rand(1, self.episode_length*self.num_episodes) # used to store reward for each time step
    self.my_episodic_cumulative_reward_log = np.random.rand(1,self.num_episodes)
    self.update_q_label = 0
    self.color_array = ['blue','orange']
    self.episode_counter = 0
    self.my_exploit_action_log = np.random.rand(self.gridnum,self.gridnum)
    self.my_state_log = np.random.rand(2, self.episode_length*self.num_episodes)
    pass
  def render(self,fig,ax,time_index): #mode='human', close=False <- no idea what this is for
    ax = fig.gca()    
    ax.clear()
    ax.grid(which='major', axis='both', linestyle='-')
    circle2 = plt.Circle((world.location_x+0.5, world.location_y+0.5), 0.5, color=self.color_array[np.mod(self.episode_counter,2)])#rand initialization
    ax.add_artist(circle2)
    fig.canvas.draw()
    pass
  def render_init(self):
    fig, (ax)=plt.subplots()
    intervals = float(1/world.gridnum)# dimension of grid affects size
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_major_locator(loc)
    ax.set_xlim(0, world.gridnum)
    ax.yaxis.set_major_locator(loc)
    ax.set_ylim(0, world.gridnum)
    return fig, ax
  def reset(self):
    self.location_x = np.random.randint(int(1),self.gridnum-1)
    self.location_y = np.random.randint(int(1),self.gridnum-1)
    self.previous_x = self.location_x
    self.previous_y = self.location_y
    self.previous_previous_x = self.previous_x
    self.previous_previous_y = self.previous_y
    self.update_q_label = 0
    self.episode_counter += 1
    return self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y, self.update_q_label, self.episode_counter
  def step(self, my_reward, desired_action, location_x, location_y, previous_x, previous_y, previous_previous_x, previous_previous_y):
    (self.my_reward, self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y) = (my_reward, location_x, location_y, previous_x, previous_y, previous_previous_x, previous_previous_y)
    self.previous_previous_x = self.previous_x
    self.previous_previous_y = self.previous_y
    self.previous_x = self.location_x# only accurate for second step in episode
    self.previous_y = self.location_y# only accurate for second step in episode
#    desired_action = self.action
    self.action = desired_action
    if np.random.rand() <= 0.1: # this part of the method is to enforce a 10% chance of a random transition
        self.action = self.allowed_actions[0,np.random.randint(0,self.allowed_actions.shape[1])]#think the 1 should be changed to a 0 for the first argument of randint
    if self.my_reward[self.location_y,self.location_x] > 5:
        self.location_x = 1+1
        self.location_y = 0+1
    elif self.my_reward[self.location_y,self.location_x] > 0:
        self.location_x = 3+1
        self.location_y = 2+1
    elif self.action == 1: # this part of the method is to select the desired deterministic action
        self.location_y += 1
    elif self.action == 2:
        self.location_y += -1
    elif self.action == 3:
        self.location_x += -1
    elif self.action == 4:
        self.location_x += 1
    return (self.location_x, self.location_y, self.previous_x, self.previous_y, self.previous_previous_x, self.previous_previous_y)
  def work(self):
   if world.render_label == 't':
       fig, (ax) = world.render_init()
   k=0 # counter for episodic cumulative reward
   for i in range(1,world.episode_length * world.num_episodes - 1):
#    world.update_reward_model()
       world.my_reward_log[0,i] = world.my_reward[world.location_y,world.location_x]
       world.my_state_log[:,i] = np.array([world.location_x,world.location_y])[np.newaxis]
#    world.update_my_q_function()#update is for previous state, so put before state reversion
       world.update_q_label += 1# default is to update the q function after the first iteration
       if world.my_reward[world.location_y,world.location_x] < 0:
           world.location_x = world.previous_x
           world.location_y = world.previous_y
#    world.my_policy() # closest to pragmatic results here
       world.step()# current state is now state AFTER action has been taken
       if np.mod(i+1,world.episode_length) == 0:
           world.my_episodic_cumulative_reward_log[0,k] = \
           np.sum(world.my_reward_log[0,(k*world.episode_length):(i+1)])# sums from k*episode_length to i
           k += 1
           world.reset()
       progress_checker = np.floor(0.1*world.episode_length*world.num_episodes)
       if np.mod(i+1,progress_checker) == 0:
           sys.stdout.write("\r"+"%s" % int(10+np.floor(i/progress_checker)*10) + '%')#updates progress without excessive output
       if world.render_label == 't':
           world.render(fig,ax,i)
   sys.stdout.write("\r"+'done' + '\n')#displays complete progress and prints results on new lines
   fig1, (ax1)=plt.subplots()
   ax1.plot(world.my_episodic_cumulative_reward_log[0,0:-1])
   plt.xlabel('episode number')
   plt.ylabel('total episodic reward')
   print('reward model' + '\n' + str(np.flipud(world.my_reward_model[1:6,1:6])))
   print('Q function for up action' + '\n' + str(np.flipud(np.transpose(world.my_q_function[1:6,1:6,0]))))
   print('Q function for down action' + '\n' + str(np.flipud(np.transpose(world.my_q_function[1:6,1:6,1]))))
   print('Q function for left action' + '\n' + str(np.flipud(np.transpose(world.my_q_function[1:6,1:6,2]))))
   print('Q function for right action' + '\n' + str(np.flipud(np.transpose(world.my_q_function[1:6,1:6,3]))))
   print('total reward = ' + str(np.sum(world.my_reward_log[0,-world.episode_length:-1])))
   fig2, (ax2) = plt.subplots()
   ax2.plot(world.my_reward_log[0,-world.episode_length:-1])
   plt.xlabel('time step of episode')
   plt.ylabel('reward')
   fig3, (ax3) = plt.subplots()
   ax3.plot(np.transpose(world.my_state_log[0,-world.episode_length:-1]), label='x coordinate')
   ax3.plot(np.transpose(world.my_state_log[1,-world.episode_length:-1]), label='y coordinate')
   plt.xlabel('time step of episode')
   plt.ylabel('x and y coordinates')
   pylab.legend(loc='upper left')
   print('Exploit policy of agent, where: 1 is up, 2 is down, 3 is left and 4 is right')
   print(np.flipud(world.my_exploit_action_log[1:6,1:6]).astype(int))
   pass
#world = gridworld()
