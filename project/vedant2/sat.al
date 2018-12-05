% MotionGenesis file:  MGSpinStability3DRigidBodyWithQuaternion.txt
% Copyright (c) 2009 Motion Genesis LLC.  All rights reserved.
%------------------------------------------------------------
NewtonianFrame  N
RigidBody       B
%-------------------------------------------------------------------
Variable   e0',  e1',  e2',  e3'   % Euler parameters (quaternion).
Variable   wx',  wy',  wz'         % Angular velocity measures.
B.SetInertia( Bcm,  Ixx = 1 kg*m^2,  Iyy = 2 kg*m^2,  Izz = 3 kg*m^2 )
%-------------------------------------------------------------------
%       Rotational kinematics and kinematical ODEs.
B.SetAngularVelocityAcceleration( N,  wx*Bx> + wy*By> + wz*Bz> )
B.SetRotationMatrixODE( N, Quaternion, e0, e1, e2, e3 )
%-------------------------------------------------------------------
%       Form equations of motion (angular momentum principle).
Dynamics[1] = Dot(  Bx>,  B.GetDynamics(Bcm)  )
Dynamics[2] = Dot(  By>,  B.GetDynamics(Bcm)  )
Dynamics[3] = Dot(  Bz>,  B.GetDynamics(Bcm)  )
Solve( Dynamics = 0,   wx',  wy',  wz'  )
%-----------------------------------------------------------
%       Numerical integration parameters and initial values.
Input  tFinal = 8 sec,  tStep = 0.01 sec, absError = 1.0E-7
Input  e0 = 1 noUnits,  e1 = 0 noUnits,  e2 = 0 noUnits,  e3 = 0 noUnits
Input  wx = 0.2 rad/sec,  wy = 7.0 rad/sec,  wz = 0.2 rad/sec
%------------------------------------------------------------
%       List output quantities and solve ODEs.
theta = GetAngleBetweenUnitVectors( Ny>,  By> )
OutputPlot  t sec,  theta degrees
ODE() MGSpinStability3DRigidBodyWithQuaternion
%------------------------------------------------------------
Save MGSpinStability3DRigidBodyWithQuaternion.html
Quit