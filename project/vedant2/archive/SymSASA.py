# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 01:05:34 2018

@author: Vedant
"""

import pybullet as p
import time
import numpy as np
    

p.connect(p.GUI)
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0,0)


#Sat part shapes
Sat_body = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2.0  , 2.3, 3.0])
Panel_right = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2, 0.2, 3])
Panel_left = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2, 0.2, 3])

#The with other shapes linked to it
body_Mass = 500
visualShapeId = -1
link_Masses=[10, 10, 10, 10]



linkCollisionShapeIndices=[Panel_right, Panel_right, Panel_left, Panel_left]
nlnk=len(link_Masses)
linkVisualShapeIndices=[-1]*nlnk    #=[-1,-1,-1, ... , -1]
#link positions wrt the link they are attached to
BasePositionR_body = np.array([4.2,0,0])
BasePositionR_panel = np.array([4.2,0,0])
BasePositionL_body = np.array([-4.2,0,0])
BasePositionL_panel = np.array([-4.2,0,0])

linkPositions=[BasePositionR_body, BasePositionR_panel, BasePositionL_body, BasePositionL_panel,]
linkOrientations=[[0,0,0,1]]*nlnk
#linkInertialFramePositions=[[-2,0,0]]*nlnk

linkInertialFramePositions = [BasePositionR_body, BasePositionR_body+BasePositionR_panel, BasePositionL_body, BasePositionL_body+BasePositionL_panel,]

print(linkInertialFramePositions)
#Note the orientations are given in quaternions (4 params). There are function to convert of Euler angles and back
linkInertialFrameOrientations=[[0,0,0,1]]*nlnk
#indices determine for each link which other link it is attached to
# for example 3rd index = 2 means that the front left knee jjoint is attached to the front left hip
indices=[0, 1, 0, 3]
#Most joint are revolving. The prismatic joints are kept fixed for now
jointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]# JOINT_SPHERICAL,JOINT_REVOLUTE
#revolution axis for each revolving joint
axis=[[0,0,1], [0,0,1], [0,0,1], [0,0,1]]

#Drop the body in the scene at the following body coordinates
basePosition = [0,0,4]
baseOrientation = [0,0,0,1]
#Main function that creates the dog
sat = p.createMultiBody(body_Mass,Sat_body,visualShapeId,basePosition,baseOrientation,
                        linkMasses=link_Masses,
                        linkCollisionShapeIndices=linkCollisionShapeIndices,
                        linkVisualShapeIndices=linkVisualShapeIndices,
                        linkPositions=linkPositions,
                        linkOrientations=linkOrientations,
                        linkInertialFramePositions=linkInertialFramePositions,
                        linkInertialFrameOrientations=linkInertialFrameOrientations,
                        linkParentIndices=indices,
                        linkJointTypes=jointTypes,
                        linkJointAxis=axis		)#	

#Add earth like gravity
p.setGravity(0,0,0)
p.resetDebugVisualizerCamera( cameraDistance=20., cameraYaw=15, cameraPitch=-65, cameraTargetPosition=[4., 4., -1.])

joint=1
p.setJointMotorControl2(sat,joint,p.POSITION_CONTROL,targetPosition=1.1,force=1,maxVelocity=3)
#Same for the prismatic feet spheres
joint=1
p.setJointMotorControl2(sat,joint,p.POSITION_CONTROL,targetPosition=1.1,force=1,maxVelocity=3)
joint=2
p.setJointMotorControl2(sat,joint,p.POSITION_CONTROL,targetPosition=1.1,force=1,maxVelocity=3)
joint=3
p.setJointMotorControl2(sat,joint,p.POSITION_CONTROL,targetPosition=1.1,force=1,maxVelocity=3)
#joint=4
p.setJointMotorControl2(sat,joint,p.POSITION_CONTROL,targetPosition=1.1,force=1,maxVelocity=3)

p.setRealTimeSimulation(1)
time.sleep(100000)
'''



p.setRealTimeSimulation(1)
#Point the camera at the robot at the desired angle and distance
p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.25])

boxHalfLength = 2.5
boxHalfWidth = 2.5
boxHalfHeight = 0.2
sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
mass = 1
block=p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                        basePosition = [-2,0,-0.1],baseOrientation=[0.0,0.1,0.0,1])
'''