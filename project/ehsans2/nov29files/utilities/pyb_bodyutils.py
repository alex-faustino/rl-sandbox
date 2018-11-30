import pybullet as p
import numpy as np

def create_side_sliders(slide_height = 1, slide_width = 0.1,
                        wall_thickness = 0.04, centerPosition = [0,0,0.52], 
                        left_wall = False, right_wall = False, physicsClientId = 0):

    half_extents = [[wall_thickness/2, wall_thickness/2, slide_height/2 + wall_thickness], #Left and right walls
                    [wall_thickness/2, slide_width/2 + wall_thickness, wall_thickness/2]] #Up and Down walls
    relative_bases = [[0, slide_width/2 + wall_thickness/2, 0], #Left and right walls
                      [0, 0, slide_height/2 + wall_thickness/2]] # Up and down walls
    directions = [-1, 1]

    for half_extent, relative_base  in zip(half_extents, relative_bases):
        for direction in directions:
            basePosition = np.array(centerPosition) + np.array(relative_base) * direction
            basePosition = basePosition.tolist()
            colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents = half_extent, physicsClientId = physicsClientId)
            vizShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents = half_extent, rgbaColor=[1,0,0,0], physicsClientId = physicsClientId)
            sphereUid = p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = colShapeId, baseVisualShapeIndex = vizShapeId, 
                                          basePosition = basePosition, baseOrientation = [0,0,0,1], physicsClientId = physicsClientId)
    
    directions = []
    directions = directions + [ 1] if right_wall else directions
    directions = directions + [-1] if left_wall else directions
    for direction in directions:
        wall_center = np.array(centerPosition) + np.array([wall_thickness, 0 , 0])*direction
        wall_center = wall_center.tolist()
        half_extent = [wall_thickness/2, slide_width/2+wall_thickness, slide_height/2+wall_thickness]
        colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents = half_extent, physicsClientId = physicsClientId)
        vizShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents = half_extent, rgbaColor=[1,0,0,0], physicsClientId = physicsClientId)
        sphereUid = p.createMultiBody(baseMass = 0, baseCollisionShapeIndex = colShapeId, baseVisualShapeIndex = vizShapeId, 
                                      basePosition = wall_center, baseOrientation = [0,0,0,1], physicsClientId = physicsClientId)

def deactivate_joint_motors(robot_unqid, joint_list = None, physicsClientId = 0, p=p):
    if joint_list is None:
        joint_list = range(p.getNumJoints(robot_unqid, physicsClientId = physicsClientId)) 
    for joint_id in joint_list:
        p.setJointMotorControl2(bodyUniqueId=robot_unqid, 
                                jointIndex=joint_id, 
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=0,
                                force = 0,
                                physicsClientId = physicsClientId)