# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:21:13 2018

@author: vedant2
"""

from sympy import symbols
from sympy.physics.mechanics import *


'''
World Frame Defination
'''
W = ReferenceFrame('W')
O = Point('O')
O.set_vel(W,0)

'''
----------------------------------------------------------------------------------------------------------------------------------------------------------
Main Body Frame Defination
----------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Body_quaternion = dynamicsymbols('q:4')
Body_omega = dynamicsymbols('w:3')
Body_vel = dynamicsymbols('v:3') 
Body_position = dynamicsymbols('x:3')

Body_frame = W.orientnew('Body', 'Quaternion', Body_quaternion)
Body_frame.set_ang_vel(W, Body_omega[0]*W.x+Body_omega[1]*W.y+Body_omega[2]*W.z)

Body_COM = Point('Body_COM')
Body_COM.set_pos(O,Body_position[0]*W.x+Body_position[1]*W.y+Body_position[2]*W.z)
Body_COM.set_vel(W,Body_vel[0]*W.x+Body_vel[1]*W.y+Body_vel[2]*W.z)

Body_connectionR = Point('Body_connectionR')
Body_connectionR.set_pos(Body_COM,2*Body_frame.x+3*Body_frame.y+4*Body_frame.z)
Body_connectionR.set_vel(Body_frame,0*Body_frame.x+0*Body_frame.y+0*Body_frame.z)
Body_connectionR.v2pt_theory(Body_COM,W,Body_frame)

Body_connectionL = Point('Body_connectionL')
Body_connectionL.set_pos(Body_COM,-2*Body_frame.x-3*Body_frame.y-4*Body_frame.z)
Body_connectionL.set_vel(Body_frame,0*Body_frame.x+0*Body_frame.y+0*Body_frame.z)
Body_connectionL.v2pt_theory(Body_COM,W,Body_frame)
#Body_frame.set_ang_vel(Body_frame, 0*Body_frame.x+0*Body_frame.y+0*Body_frame.z)
KDE_Body = []

#Body_frame.ang_vel_in(W)
#Body_COM.v2pt_theory(O,W,Body_frame)
# missing linear vel!



I1_Body, I2_Body, I3_Body, M_Body = symbols('I1_Body I2_Body I3_Body M_Body')
Body_I = (inertia(Body_frame, I1_Body, I2_Body, I3_Body, 0, 0,0))
Body = RigidBody('Body', Body_COM, Body_frame, M_Body, (Body_I, Body_COM))

Torque_Base_PR1_x, Torque_Base_PR1_y, Torque_Base_PR1_z = dynamicsymbols('Torque_Base_PR1_x Torque_Base_PR1_y Torque_Base_PR1_z')
Torque_Base_PL1_x, Torque_Base_PL1_y, Torque_Base_PL1_z = dynamicsymbols('Torque_Base_PL1_x Torque_Base_PL1_y Torque_Base_PL1_z')

Body_torque = (-Torque_Base_PR1_x-Torque_Base_PL1_x)*Body_frame.x+(-Torque_Base_PR1_y-Torque_Base_PL1_y)*Body_frame.y+ (-Torque_Base_PR1_z-Torque_Base_PL1_z)*Body_frame.z


'''
----------------------------------------------------------------------------------------------------------------------------------------------------------
First Right Panel Frame
----------------------------------------------------------------------------------------------------------------------------------------------------------
'''

K_PanelR1_x, K_PanelR1_y, K_PanelR1_y = symbols('K_PanelR1_x K_PanelR1_y K_PanelR1_y')

PanelR1_theta = dynamicsymbols('PR1_th:3')
PanelR1_omega = dynamicsymbols('PR1_om:3')

PanelR1_frame = W.orientnew('PR1', 'Body', PanelR1_theta , '123')
PanelR1_frame.set_ang_vel(Body_frame, PanelR1_omega[0]*Body_frame.x+PanelR1_omega[1]*Body_frame.y+PanelR1_omega[2]*Body_frame.z)

PanelR1_O = Point('PR1_O')
PanelR1_O.set_pos(Body_connectionR,0*PanelR1_frame.x+0*PanelR1_frame.y+0*PanelR1_frame.z)# Body_frame or PanelR1_frame?
PanelR1_O.set_vel(PanelR1_frame,0*PanelR1_frame.x+0*PanelR1_frame.y+0*PanelR1_frame.z)

PanelR1_COM = Point('PR1_COM')
PanelR1_COM.set_pos(PanelR1_O,1*PanelR1_frame.x+2*PanelR1_frame.y+3*PanelR1_frame.z)
PanelR1_COM.set_vel(PanelR1_frame,0*PanelR1_frame.x+0*PanelR1_frame.y+0*PanelR1_frame.z)
PanelR1_COM.v2pt_theory(Body_connectionR,Body_frame,PanelR1_frame)

KDE_PanelR1 = [PanelR1_theta[0].diff() - PanelR1_omega[0],
       PanelR1_theta[1].diff() - PanelR1_omega[1],
       PanelR1_theta[2].diff() - PanelR1_omega[2]]

I1_PanelR1, I2_PanelR1, I3_PanelR1, M_PanelR1 = symbols('I1_PanelR1 I2_PanelR1 I3_PanelR1 M_PanelR1')
PanelR1_I = (inertia(PanelR1_frame, I1_PanelR1, I2_PanelR1, I3_PanelR1, 0, 0, 0))
PanelR1 = RigidBody('PanelR1', PanelR1_COM, PanelR1_frame, M_PanelR1, (PanelR1_I, PanelR1_COM))
PanelR1_torque = (Torque_Base_PR1_x)*PanelR1_frame.x+(Torque_Base_PR1_y)*PanelR1_frame.y+ (Torque_Base_PR1_z)*PanelR1_frame.z

PanelR1.potential_energy = 0.5*(K_PanelR1_x*PanelR1_theta[0]**2+K_PanelR1_y*PanelR1_theta[1]**2+K_PanelR1_y*PanelR1_theta[2]**2)

'''
----------------------------------------------------------------------------------------------------------------------------------------------------------
First Left Panel Frame
----------------------------------------------------------------------------------------------------------------------------------------------------------
'''

K_PanelL1_x, K_PanelL1_y, K_PanelL1_y = symbols('K_PanelL1_x K_PanelL1_y K_PanelL1_y')

PanelL1_theta = dynamicsymbols('PL1_th:3')
PanelL1_omega = dynamicsymbols('PL1_om:3')

PanelL1_frame = W.orientnew('PL1', 'Body', PanelL1_theta , '123')
PanelL1_frame.set_ang_vel(Body_frame, PanelL1_omega[0]*Body_frame.x+PanelL1_omega[1]*Body_frame.y+PanelL1_omega[2]*Body_frame.z)

PanelL1_COM = Point('PL1_COM')
PanelL1_COM.set_pos(Body_connectionL,1*PanelL1_frame.x+2*PanelL1_frame.y+3*PanelL1_frame.z)
PanelL1_COM.set_vel(PanelL1_frame,0*PanelL1_frame.x+0*PanelL1_frame.y+0*PanelL1_frame.z)
PanelL1_COM.v2pt_theory(Body_connectionL,Body_frame,PanelL1_frame)

KDE_PanelL1 = [PanelL1_theta[0].diff() - PanelL1_omega[0],
       PanelL1_theta[1].diff() - PanelL1_omega[1],
       PanelL1_theta[2].diff() - PanelL1_omega[2]]

I1_PanelL1, I2_PanelL1, I3_PanelL1, M_PanelL1 = symbols('I1_PanelL1 I2_PanelL1 I3_PanelL1 M_PanelL1')
PanelL1_I = (inertia(PanelL1_frame, I1_PanelL1, I2_PanelL1, I3_PanelL1, 0, 0, 0))
PanelL1 = RigidBody('PanelL1', PanelL1_COM, PanelL1_frame, M_PanelL1, (PanelL1_I, PanelL1_COM))
PanelL1_torque = (Torque_Base_PL1_x)*PanelL1_frame.x+(Torque_Base_PL1_y)*PanelL1_frame.y+ (Torque_Base_PL1_z)*PanelL1_frame.z

PanelL1.potential_energy = 0.5*(K_PanelL1_x*PanelL1_theta[0]**2+K_PanelL1_y*PanelL1_theta[1]**2+K_PanelL1_y*PanelL1_theta[2]**2)


Lagrangian(W, Body, PanelR1, PanelL1)
