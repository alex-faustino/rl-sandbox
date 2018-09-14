# -*- coding: utf-8 -*-
# Python Space Simulator v 0.0.1
# Test ADCS control

# notes
# - variables with _data are the collection of all data for that variable,
#   before I used bucket
# - scalars are lower cap, vector and matrices are all CAPS

# system modules
import os
import sys
from time import time as Time
from time import sleep
tic = Time()

from modules import *
from math import pow, degrees, radians, pi, sin, cos, sqrt

import pdb; 

# ------------------------------------------------------------------------------
# GLOBAL CONSTANTS
inRadians = pi/180.0; # conversion to radians
inDegrees = 180.0/pi; # conversion to radians
mu = 3.986004e14;     # m J /kg
r_earth = 6378.14e3;  # Earth Radius [m]

# ------------------------------------------------------------------------------
# CONTROL VARIABLES
nOrbits = 0.1; #number of orbits
dt      = 1.0; # time step
model = 'hiakasat';

#-------------------------------------------------------------------------------
# INITIALIZATION
# initial angular speed [deg/s]

# euler sequence: yaw, pitch, roll
# 
euler_0 = array([100,0,0]) * inRadians;

# conver euler angles to quaternions
# q is defined with scalar first [q0 q1 q2 q3]
q_0 = quaternion_from_euler(euler_0);

# 
omega_0 = array([0.0, 0.0, 1.0]) * inRadians;

# orbital data
h_sat = 500e3; # height [m]
r_sat = r_earth + h_sat; # radius [m]
v = sqrt(mu/r_sat); # speed [m/s]
P = 2*pi*r_sat/v; #period [s]
Pminutes = P/60;

tf = nOrbits*P;#sec
#tf = 1000;

# for gravity gradient stability
# Iy > Ix > Iz
I_xx=2.5448;
I_yy=2.4444;
I_zz=2.6052;

#I_xx=1;
#I_yy=1;
#I_zz=1;

Inertia =  diag([I_xx,I_yy,I_zz]);


##
# state vector
X_0 = concatenate([q_0,omega_0]);
#t  = 0:dt:tf;                                   
# set time points
time_data = arange(0,tf,dt)

Torque_0 = array([0.0, 0.0 , 0.0]);


# initialize data buckets with zeros
X_data              = zeros([alen(time_data),7]);
attitude_error_data = zeros([alen(time_data),3]);
Torque_data         = zeros([alen(time_data),3]);

# assign initial value for simulation
X = X_0;
Torque = Torque_0



# desired attitude
euler_ref = array([0.0,0.0,0.0]);

error_last = array([0.0,0.0,0.0]);

# -----------------------------------------------------------------------------
# start attitude propagation simulation

#X_data = integrate.odeint(attitude_dynamics, X_0, time_data, args=(Torque,Inertia))
    
for k in enumerate(time_data):
    #print time_data[k]
    
    
    # compute error
    error = euler_ref - euler
    derror = (error-error_last)/dt
    error_last = error
    
    # for symetric Inertia = 1
    #kp = 0.01
    #kd = 0.1
    
    # for hiakasat Inertia
    kp = 0.001
    kd = 0.1
    
    tx = kp*error[2] + kd*derror[2]
    ty = kp*error[1] + kd*derror[1]
    tz = kp*error[0] + kd*derror[0]
    
    Torque = array([tx,ty,tz])
    Torque_data[k[0],:] = Torque
    

# assign
q_data     = X_data[:,0:4];
omega_data = X_data[:,4:7];

# convert quaternions to euler
euler_data = zeros([alen(time_data),3]);
for k in  enumerate(time_data):
   euler_data[k[0],:] = euler_from_quaternion(q_data[k[0],:]);

#-------------------------------------------------------------------------------
# INIT FIGURE
#-------------------------------------------------------------------------------

#turn interactive mode on
ion()

# to get size: fig.get_size_inches()
fig = plt.figure(1) #figsize=(6.5, 9.5))

#clear the figure
fig.clf()
#fig.canvas.manager.window.move(900,0) # in pixels from top-left corner

#ax = fig.add_subplot(111)

time_data = time_data/60

# plot measurements and compare with estimated fiter measurements
plt.subplot(311)
plt.plot(time_data, q_data)
legend(['$q_0$','$q_1$','$q_2$','$q_3$'])
ylim(-1.1, 1.1)

#legend(['$\omega_x$','$\omega_y$','$\omega_z$'])
#ylabel('$\Omega$ [deg/sec]')

plt.subplot(312)
plt.plot(time_data, euler_data*180/pi) 
legend(['$\\psi (yaw)$','$\\theta (pitch)$','$\phi (roll)$'])
ylim(-200, 200)

plt.subplot(313)
plt.plot(time_data, Torque_data) 
legend(['$T_x$','$T_y$','$T_z$'])
ylim(-0.200, 0.0200)

# to redraw
plt.show()
#-------------------------------------------------------------------------------
# ELAPSED TIME
#-------------------------------------------------------------------------------
toc = Time()

print( "Elapsed time: {:.3f} s".format(toc-tic))