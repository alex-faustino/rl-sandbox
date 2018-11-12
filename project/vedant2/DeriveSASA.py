# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:21:13 2018

@author: vedant2
"""

from sympy import symbols
from sympy.physics.mechanics import *



W = ReferenceFrame('W')
O = Point('O')
O.set_vel(W,0)

Body_quaternion = dynamicsymbols('q:4')
Body_omega = dynamicsymbols('w:3')
Body_vel = dynamicsymbols('v:3') 
Body_position = dynamicsymbols('x:3')
Body_frame = W.orientnew('B', 'Quaternion', Body_quaternion)
I1, I2, I3, M_base = symbols('I1 I2 I3 M_Base')
B_Frame_I = (inertia(B, I1, I2, I3, 0, 0,0))
Body_COM = Point('Body_COM')
Body_COM.set_pos(O,Body_position[0]*W.x+Body_position[1]*W.y+Body_position[2]*W.z)
Base = RigidBody('Base', Body_COM, W, M_base, (B_Frame_I, Body_COM))

B_connect = Point('B_connect')
B_connect.set_pos(Body_COM,2*Body_frame.x+3*Body_frame.y+4*Body_frame.z)
KDE_B = []
Body_frame.set_ang_vel(W, Body_omega[0]*W.x+Body_omega[1]*W.y+Body_omega[2]*W.z)
Body_frame.ang_vel_in(W)
#Body_COM.v2pt_theory(O,W,Body_frame)
# missing linear vel!
Body_COM.set_vel(W,Body_vel[0]*W.x+Body_vel[1]*W.y+Body_vel[2]*W.z)

PanelR1_theta = dynamicsymbols('PR1_th:3')
Panel_omega = dynamicsymbols('PR1_om:3')
PanelR1_frame = W.orientnew('PR1', 'Body', PanelR1_theta , '123')
PanelR1_connection = Point('PR1_O')
PanelR1_connection.set_pos(B_connect,0*PanelR1_frame.x+0*PanelR1_frame.y+0*PanelR1_frame.z)
PanelR1_COM = Point('PR1_COM')
PanelR1_COM.set_pos(PanelR1_connection,1*PanelR1_frame.x+2*PanelR1_frame.y+3*PanelR1_frame.z)

KDE_R1 = [PanelR1_theta[0].diff() - Panel_omega[0],
       PanelR1_theta[1].diff() - Panel_omega[1],
       PanelR1_theta[2].diff() - Panel_omega[2]]

PanelR1_frame.set_ang_vel(Body_frame, Panel_omega[0]*Body_frame.x+Panel_omega[1]*Body_frame.y+Panel_omega[2]*Body_frame.z)
PanelR1_frame.ang_vel_in(Body_frame)
PanelR1_COM.v2pt_theory(B_connect,Body_frame,PanelR1_frame)
'''
from sympy import symbols
import sympy.physics.mechanics as me

print("Defining the problem.")

# The conical pendulum will have three links and three bobs.
n = 3

# Each link's orientation is described by two spaced fixed angles: alpha and
# beta.

# Generalized coordinates
alpha = me.dynamicsymbols('alpha:{}'.format(n))
beta = me.dynamicsymbols('beta:{}'.format(n))

# Generalized speeds
omega = me.dynamicsymbols('omega:{}'.format(n))
delta = me.dynamicsymbols('delta:{}'.format(n))

# At each joint there are point masses (i.e. the bobs).
m_bob = symbols('m:{}'.format(n))

# Each link is modeled as a cylinder so it will have a length, mass, and a
# symmetric inertia tensor.
l = symbols('l:{}'.format(n))
m_link = symbols('M:{}'.format(n))
Ixx = symbols('Ixx:{}'.format(n))
Iyy = symbols('Iyy:{}'.format(n))
Izz = symbols('Izz:{}'.format(n))

# Acceleration due to gravity will be used when prescribing the forces
# acting on the links and bobs.
g = symbols('g')

# Now defining an inertial reference frame for the system to live in. The Y
# axis of the frame will be aligned with, but opposite to, the gravity
# vector.

I = me.ReferenceFrame('I')

# Three reference frames will track the orientation of the three links.

A = me.ReferenceFrame('A')
A.orient(I, 'Space', [alpha[0], beta[0], 0], 'ZXY')

B = me.ReferenceFrame('B')
B.orient(A, 'Space', [alpha[1], beta[1], 0], 'ZXY')

C = me.ReferenceFrame('C')
C.orient(B, 'Space', [alpha[2], beta[2], 0], 'ZXY')

# Define the kinematical differential equations such that the generalized
# speeds equal the time derivative of the generalized coordinates.
kinematic_differentials = []
for i in range(n):
    kinematic_differentials.append(omega[i] - alpha[i].diff())
    kinematic_differentials.append(delta[i] - beta[i].diff())

# The angular velocities of the three frames can then be set.
A.set_ang_vel(I, omega[0] * I.z + delta[0] * I.x)
B.set_ang_vel(I, omega[1] * I.z + delta[1] * I.x)
C.set_ang_vel(I, omega[2] * I.z + delta[2] * I.x)

# The base of the pendulum will be located at a point O which is stationary
# in the inertial reference frame.
O = me.Point('O')
O.set_vel(I, 0)

# The location of the bobs (at the joints between the links) are created by
# specifiying the vectors between the points.
P1 = O.locatenew('P1', -l[0] * A.y)
P2 = P1.locatenew('P2', -l[1] * B.y)
P3 = P2.locatenew('P3', -l[2] * C.y)

# The velocities of the points can be computed by taking advantage that
# pairs of points are fixed on the referene frames.
P1.v2pt_theory(O, I, A)
P2.v2pt_theory(P1, I, B)
P3.v2pt_theory(P2, I, C)
points = [P1, P2, P3]

# Now create a particle to represent each bob.
Pa1 = me.Particle('Pa1', points[0], m_bob[0])
Pa2 = me.Particle('Pa2', points[1], m_bob[1])
Pa3 = me.Particle('Pa3', points[2], m_bob[2])
particles = [Pa1, Pa2, Pa3]

# The mass centers of each link need to be specified and, assuming a
# constant density cylinder, it is equidistance from each joint.
P_link1 = O.locatenew('P_link1', -l[0] / 2 * A.y)
P_link2 = P1.locatenew('P_link2', -l[1] / 2 * B.y)
P_link3 = P2.locatenew('P_link3', -l[2] / 2 * C.y)

# The linear velocities can be specified the same way as the bob points.
P_link1.v2pt_theory(O, I, A)
P_link2.v2pt_theory(P1, I, B)
P_link3.v2pt_theory(P2, I, C)

points_rigid_body = [P_link1, P_link2, P_link3]

# The inertia tensors for the links are defined with respect to the mass
# center of the link and the link's reference frame.
inertia_link1 = (me.inertia(A, Ixx[0], Iyy[0], Izz[0]), P_link1)
inertia_link2 = (me.inertia(B, Ixx[1], Iyy[1], Izz[1]), P_link2)
inertia_link3 = (me.inertia(C, Ixx[2], Iyy[2], Izz[2]), P_link3)

# Now rigid bodies can be created for each link.
link1 = me.RigidBody('link1', P_link1, A, m_link[0], inertia_link1)
link2 = me.RigidBody('link2', P_link2, B, m_link[1], inertia_link2)
link3 = me.RigidBody('link3', P_link3, C, m_link[2], inertia_link3)
links = [link1, link2, link3]

# The only contributing forces to the system is the force due to gravity
# acting on each particle and body.
forces = []

for particle in particles:
    mass = particle.mass
    point = particle.point
    forces.append((point, -mass * g * I.y))

for link in links:
    mass = link.mass
    point = link.masscenter
    forces.append((point, -mass * g * I.y))

# Make a list of all the particles and bodies in the system.
total_system = links + particles

# Lists of all generalized coordinates and speeds.
q = alpha + beta
u = omega + delta

# Now the equations of motion of the system can be formed.
print("Generating equations of motion.")
kane = me.KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kinematic_differentials)
fr, frstar = kane.kanes_equations(forces, total_system)
print("Derivation complete.")
'''

from sympy import Symbol
from sympy.physics.mechanics import ReferenceFrame, Point, RigidBody
from sympy.physics.mechanics import outer
m = Symbol('m')
A = ReferenceFrame('A')
P = Point('P')
I = outer (A.x, A.x)
inertia_tuple = (I, P)
B = RigidBody('B', P, A, m, inertia_tuple)
# Or you could change them afterwards
m2 = Symbol('m2')
B.mass = m2

M, v, r, omega = symbols('M v r omega')
N = ReferenceFrame('N')
b = ReferenceFrame('b')
b.set_ang_vel(N, omega * b.x)
P = Point('P')
P.set_vel(N, v * N.x)
I = outer (b.x, b.x)
inertia_tuple = (I, P)
B = RigidBody('B', P, b, M, inertia_tuple)
B.kinetic_energy(N)



from sympy.physics.mechanics import Point, Particle, ReferenceFrame
from sympy.physics.mechanics import RigidBody, outer, Lagrangian
from sympy import symbols
M, m, g, h = symbols('M m g h')
N = ReferenceFrame('N')
O = Point('O')
O.set_vel(N, 0 * N.x)
P = O.locatenew('P', 1 * N.x)
P.set_vel(N, 10 * N.x)
Pa = Particle('Pa', P, 1)
Ac = O.locatenew('Ac', 2 * N.y)
Ac.set_vel(N, 5 * N.y)
a = ReferenceFrame('a')
a.set_ang_vel(N, 10 * N.z)
I = outer(N.z, N.z)
A = RigidBody('A', Ac, a, 20, (I, Ac))
Pa.potential_energy = m * g * h
A.potential_energy = M * g * h
Lagrangian(N, Pa, A)
