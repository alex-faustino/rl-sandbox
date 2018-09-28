from gym import core, spaces
from gym.envs.classic_control.acrobot import AcrobotEnv
import gym.envs.classic_control.acrobot as ar
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi

ACTION_SET = [-2.3,0, 2.3]

class AcroBotMEnv(AcrobotEnv):
    
    def __init__(self):
         super(AcroBotMEnv,self).__init__()
         self.dt = .2
         
         self.LINK_LENGTH_1 = 1. # [m]
         self.LINK_LENGTH_2 = 3.  # [m]
         self.LINK_MASS_1 = 1.  #: [kg] mass of link 1
         self.LINK_MASS_2 = 2.  #: [kg] mass of link 2
         self.LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
         self.LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
         self.LINK_MOI1 = 1.  #: moments of inertia for  link 1
         self.LINK_MOI2 = 1.  #: moments of inertia for link 2
         self.MAX_VEL_1 = 4 * np.pi
         self.MAX_VEL_2 = 9 * np.pi
         self.MAX_TORQUE = 5.;

         #hi = np.array([self.MAX_TORQUE])
         self.nA = 3
         self.action_space = spaces.Discrete(self.nA)
         
         
    def _physics(self,s_pp,t):
        #s_pp=[stat torque(or control)]
        self.book_or_nips = "book"
        self._dsdt(self, s_pp, t) 

    def idx_to_action(self,idx):
        return ACTION_SET[idx]

    def step(self, a):
        s = self.state
        torque = this.idx_to_action(a)
  
        # Now, augment the state with our force action so it can be passed to the physEngine
        
        s_augmented = np.append(s, torque)

        ns = ar.rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action

        ns[0] = ar.wrap(ns[0], -pi, pi)
        ns[1] = ar.wrap(ns[1], -pi, pi)
        ns[2] = ar.bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = ar.bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        delta = .5;
        reward = (ns[0] >= np.pi / 2 - delta) & (ns[0] <= np.pi / 2 + delta)
        reward &= (ns[1] >= -delta) & (ns[1] <=  delta)
            
        return (self._get_ob(), float(reward), terminal, {}) 

