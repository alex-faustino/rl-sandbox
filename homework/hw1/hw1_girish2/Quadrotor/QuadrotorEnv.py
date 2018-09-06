import numpy as np 
import matplotlib.pyplot as plt 

class Quadrotor():

    def __init__(self):
        self.gravity = 9.8
        self.Ix = 0.25
        self.Iy = 0.25
        self.Iz = 0.05
        self.m = 1
        self.dt = 0.005

        # Control Limit
        self.max_control = 2

        # Speed Limit
        self.max_acceleration = 10

        self.a1 = (self.Iy - self.Iz)/ self.Ix
        self.a2 = (self.Iz - self.Ix)/ self.Iy
        self.a3 = (self.Ix - self.Iy)/ self.Iz

        self._reset()

        self.render_in = True # For plotting

    def step(self,u):
        # RK4 Update
        curr_state = self.state

        # 1st Update
        xp = self._dynamics(curr_state,u)
        rk1 = self.dt*xp
        next_state = curr_state + rk1

        # 2nd Update
        xp = self._dynamics(next_state,u)
        rk2 = self.dt*xp
        next_state = curr_state + 2.*rk2

        # 3rd Update
        xp = self._dynamics(next_state,u)
        rk3 = self.dt*xp
        next_state = curr_state + 2*rk3

        # 4th Update
        xp = self._dynamics(next_state,u)
        rk4 = self.dt*xp
        next_state = curr_state + rk4

        next_state = curr_state + (rk1 + rk2 + rk3 + rk4)/6

        self.state = next_state

        reward = self._reward(next_state)
        
        return next_state, reward
    
    def _reward(self, state):
        reward = np.exp(-np.linalg.norm(state))
        return reward
            
    def _dynamics(self,state,u):

        #Outer Loop states
        x1 = state[0]
        x2 = state[1]
        y1 = state[2]
        y2 = state[3]
        z1 = state[4]
        z2 = state[5]
        # Innner Loop States
        phi = state[6]
        theta = state[7]
        psi = state[8]
        p = state[9]
        q = state[10]
        r = state[11]
        # Control 
        U1 = u[0]
        U2 = u[1]
        U3 = u[2]
        U4 = u[3]

        # Clip Control
        #U1 = np.clip(U1, 9, 10.5)
        #U2 = np.clip(U2, -2.5e-3, 2.5e-3)
        #U3 = np.clip(U3, -2.5e-3, 2.5e-3)
        #U4 = np.clip(U4, -2.5e-3, 2.5e-3)

        x1Dot = x2
        x2Dot = (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi))*(U1/self.m)
        y1Dot = y2
        y2Dot = (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi))*(U1/self.m)
        z1Dot = z2
        z2Dot = -self.gravity + np.cos(phi)*np.cos(theta)*(U1/self.m)

        phiDot = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        thetaDot = q*np.cos(phi) - r*np.sin(phi)
        psiDot = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)

        pDot = self.a1*r*q + U2/self.Ix
        qDot = self.a2*r*p + U3/self.Iy
        rDot = self.a3*p*q + U4/self.Iz

        #pDot = np.clip(pDot, -self.max_acceleration, self.max_acceleration)
        #qDot = np.clip(qDot, -self.max_acceleration, self.max_acceleration)
        #rDot = np.clip(rDot, -self.max_acceleration, self.max_acceleration)

        XDot = np.reshape([x1Dot, x2Dot, y1Dot, y2Dot, z1Dot, z2Dot, phiDot, thetaDot, psiDot, pDot, qDot, rDot],(1,12))

        return XDot[0]

    def action_space_sample(self):
        u1 = [np.random.uniform(low=9, high=10.5, size=None)]
        u2 = [np.random.uniform(low=-2.5e-3, high=2.5e-3, size=None)]
        u3 = [np.random.uniform(low=-2.5e-3, high=2.5e-3, size=None)]
        u4 = [np.random.uniform(low=-2.5e-3, high=2.5e-3, size=None)]
        action = np.reshape([u1,u2,u3,u4],(1,4))
        return action[0]

    def _reset(self):
        self.state = np.zeros(shape=(1,12),dtype=float)[0]
        #self.state[4] = 1.
        self.state[6] = np.deg2rad(10)
        return self.state

    def render(self):

        x1r = self.state[0]- 0.5*np.cos(self.state[6])
        y1r = self.state[4]- 0.5*np.sin(self.state[6])
        x2r = self.state[0]+ 0.5*np.cos(self.state[6])
        y2r = self.state[4]+ 0.5*np.sin(self.state[6])

        x1p = self.state[0]- 0.5*np.cos(self.state[7])
        y1p = self.state[4]- 0.5*np.sin(self.state[7])
        x2p = self.state[0]+ 0.5*np.cos(self.state[7])
        y2p = self.state[4]+ 0.5*np.sin(self.state[7])

        if self.render_in:
            self.render_in = False
            plt.ion()
            self.fig = plt.figure()
            plt.title('Quadrotor Model')
            self.ax_roll = self.fig.add_subplot(121)
            self.ax_roll.axis([-self.state[0]-1,self.state[0]+1,self.state[4]-2,self.state[4]+2])
            self.line1, = self.ax_roll.plot([x1r,x2r],[y1r, y2r], marker='o')
            self.ax_pitch = self.fig.add_subplot(122)
            self.ax_pitch.axis([-self.state[0]-1,self.state[0]+1,self.state[4]-2,self.state[4]+2])
            self.line2, = self.ax_pitch.plot([x1p,x2p],[y1p, y2p], marker='o')

        else:
            self.ax_roll.axis([-self.state[0]-1,self.state[0]+1,self.state[4]-2,self.state[4]+2])
            self.ax_pitch.axis([-self.state[0]-1,self.state[0]+1,self.state[4]-2,self.state[4]+2])
            self.line1.set_xdata([x1r,x2r])
            self.line1.set_ydata([y1r, y2r])
            self.line2.set_xdata([x1p,x2p])
            self.line2.set_ydata([y1p, y2p])

        self.fig.canvas.draw()
        plt.pause(0.00000001)
        





