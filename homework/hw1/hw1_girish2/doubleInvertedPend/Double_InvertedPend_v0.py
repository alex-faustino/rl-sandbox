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
        self.m2 = 1.5
        # Calculate the inertia of the two arms of acrobot
        self.I1 = (self.m1*0.5)*(self.L1*0.5)**2
        self.I2 = (self.m2*0.5)*(self.L2*0.5)**2
        # Time step
        self.dt = 0.05
        # control Limit
        self.max_torque = 0.1
        # Speed Limit
        self.max_speed = 8
        # Reset
        self._reset()
        # Render State
        self.render_in = True
        
    def step(self,u):
        u = np.clip(u, -self.max_torque, self.max_torque)
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
        B = np.reshape([[0],[1.]],(2,1))

        thdot_dot = np.matmul(np.linalg.inv(M), (-np.matmul(C,thdot) + T + B*u))

        thdot_dot[0,0] = np.clip(thdot_dot[0,0], -self.max_speed, self.max_speed)
        thdot_dot[1,0] = np.clip(thdot_dot[1,0], -self.max_speed, self.max_speed)

        th = th + self.dt*thdot

        thdot = thdot + self.dt*thdot_dot

        self.state = [th[0,0], thdot[0,0], th[1,0], thdot[1,0]]

        cost = self._cost()

        return self.state, cost

    def action_space_sample(self):
        action = np.random.uniform(low=-self.max_torque, high=self.max_torque, size=None)
        return action

    def _cost(self):

        cost = 0 
        if (self.state[0] >= np.pi/2-0.1 and self.state[0] <= np.pi/2 + 0.1):
            if (self.state[2] >= -0.1 and self.state[2] <= 0.1):
                cost = 1
        
        return cost

    def _reset(self):
        high = np.array([np.pi, 1, np.pi, 1])
        self.state = np.random.uniform(low = -high, high= high, size=None)

    def _render(self):

        x1 = self.L1*np.cos(self.state[0])
        y1 = self.L1*np.sin(self.state[0])
        x2 = x1+self.L2*np.cos(self.state[2])
        y2 = y1+self.L2*np.sin(self.state[2])

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







 
