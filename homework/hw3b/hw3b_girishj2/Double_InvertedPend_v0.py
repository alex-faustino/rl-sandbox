'''
_______________________________________
Author: Girish Joshi (girishj2@illinois.edu)
Date:  09/02/2018

This code implements the double Inverted Pendulum Environment
Double Inverted Pendulum dynamics is taken from the article
underactuated.mit.edu/underactuated.html
_______________________________________
'''
import numpy as np 
import matplotlib.pyplot as plt

class doublePendulum():
    
    def __init__(self):

        # Define the Gravity
        self.gravity = 9.8
        # Define length of two acrobot arms
        self.L1 = 1
        self.L2 = 1
        self.L = self.L1 + self.L2 # For plotting
        # Define the mass of acrobot arms
        self.m1 = 1
        self.m2 = 1
        # Calculate the inertia of the two arms of acrobot
        self.I1 = 1
        self.I2 = 1
        # Time step
        self.dt = 0.05
        # control Limit
        self.max_torque = 5
        # Speed Limit
        self.max_speed = 5
        # Reset
        self.reset()
        # Render State
        self.render_in = True
        
    def step(self,u):
        if u == 0:
            action = -self.max_torque
        elif u == 1:
            action = 0
        elif u == 2:
            action = self.max_torque
        else:
            print('Invalid Action')
            
        
        th_1, thdot_1, th_2, thdot_2 =  self.state

        th = np.reshape([th_1, th_2],(2,1))
        thdot = np.reshape([thdot_1, thdot_2], (2,1))

        lc1 = self.L1*np.cos(th_1)
        lc2 = self.L2*np.cos(th_2)
        c1 = np.cos(th_1)
        c2 = np.cos(th_2)
        s1 = np.sin(th_1)
        s2 = np.sin(th_2)
        s12 = np.sin(th_1+th_2)
        
        M1 = self.I1+self.I2 + self.m2*(self.L1)**2 + 2*self.m2*self.L1*lc2*c2
        M2 = self.I2 + self.m2*self.L1*lc2*c2
        M3 = self.I2 + self.m2*self.L1*lc2*c2
        M4 = self.I2
        M = [M1, M2, M3, M4]

        C1 = -2*self.m2*self.L1*lc2*s2*thdot_2
        C2 = -self.m2*self.L1*lc2*s2*thdot_2
        C3 = self.m2*self.L1*lc2*s2*thdot_1
        C4 = 0
        C = [C1, C2, C3, C4]

        T1 = -self.m1*self.gravity*lc1*s1-self.m2*self.gravity*(self.L1*s1 + lc2*s12)
        T2 = -self.m2*self.gravity*lc2*s12
        T = [T1, T2]

        
        M = np.reshape(M,(2,2))
        C = np.reshape(C,(2,2))
        T = np.reshape(T,(2,1))
        B = np.reshape([[0.],[1.]],(2,1))

        #cost = -(0.1*(abs(self._angle_normalize(th_1))-np.pi)**2 + 0.05*(self._angle_normalize(th_2))**2 + 0.01*thdot_1**2 + 0.01*thdot_2**2 + 0.0001*(u-1)**2)

        thdot_dot = np.matmul(np.linalg.inv(M), (-np.matmul(C,thdot) + T + B*action))

        thdot = thdot + self.dt*thdot_dot

        th = th + self.dt*thdot     

        thdot[0,0] = np.clip(thdot[0,0], -self.max_speed, self.max_speed)
        thdot[1,0] = np.clip(thdot[1,0], -self.max_speed, self.max_speed)

        self.state = [th[0,0], thdot[0,0], th[1,0], thdot[1,0]]

        th_1, thdot_1, th_2, thdot_2 =  self.state

        # Shaped Reward Model-1
        
        # cost = -(0.1*(abs(self._angle_normalize(th_1))-np.pi)**2 + 0.05*(self._angle_normalize(th_2))**2 + 0.01*thdot_1**2 + 0.01*thdot_2**2)
        
        # Shaped Reward Model-2

        # normalized_height = (self.L1*np.cos(self._angle_normalize(th_1)+np.pi) + self.L2*np.cos(self._angle_normalize(th_1) + self._angle_normalize(th_2)+np.pi) + self.L1 + self.L2)/(2*(self.L1 + self.L2))

        # cost = np.exp(-0.5*(1-normalized_height)**2 - 0.01*thdot_1**2 - 0.01*thdot_2**2)-1

        # Shaped Reward Model-3

        cost  = self.L1*np.cos(abs(th_1) - np.pi) + self.L2*np.cos(abs(th_1 + th_2) - np.pi)

        if cost > 0:
            cost = 2*cost
        
        obs = self._observation(self.state)

        return obs, cost, False 

    def _observation(self,state):
        return [np.cos(state[0]), np.sin(state[0]), state[1], np.cos(state[2]), np.sin(state[2]), state[3]]

    def _angle_normalize(self,x):
        angle = (((x+np.pi) % (2*np.pi)) - np.pi)
        return angle

    def _rate_normalize(self,x):
        rate = x/self.max_speed
        return rate

    def action_space_sample(self):
        action = np.random.randint(low=0, high=2, size=None)
        return action

    def _cost(self):
        pass

    def reset(self):
        high = np.array([np.deg2rad(10),0.5,np.deg2rad(10), 0.5])
        self.state = np.random.uniform(low = -high, high= high, size=None)
        obs = self._observation(self.state)
        return obs

    def render(self):

        y1 = -self.L1*np.cos(self.state[0])
        x1 = self.L1*np.sin(self.state[0])
        y2 = y1-self.L2*np.cos(self.state[0]+self.state[2])
        x2 = x1+self.L2*np.sin(self.state[0]+self.state[2])

        if self.render_in:
            self.render_in = False
            plt.ion()
            self.fig = plt.figure()
            plt.axis([-self.L-0.5,self.L+0.5,-self.L-0.5,self.L+0.5])
            plt.title('Double Inverted Pendulum')
            ax = self.fig.add_subplot(111)
            self.line1, = ax.plot([0,x1],[0,y1], marker = 'o')
            self.line2, = ax.plot([x1,x2],[y1, y2], marker='o')
        else:
            self.line1.set_xdata([0,x1])
            self.line1.set_ydata([0,y1])
            self.line2.set_xdata([x1,x2])
            self.line2.set_ydata([y1, y2])

        self.fig.canvas.draw()
        plt.pause(0.00000001)
