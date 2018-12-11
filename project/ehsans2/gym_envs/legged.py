import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import subprocess, collections, time
import pybullet as p
import pybullet_data
from utilities.render_utils import compiler_video, image_renderer
from utilities.pyb_vizutils import pyb_viz
from utilities.pyb_bodyutils import create_side_sliders, deactivate_joint_motors
from PIL import Image, ImageDraw 

class robotBulletEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, gui = True, time_step = 0.0001, ground_speed = 0,
                 robot_urdf = 'urdf/mymonoped_handed.urdf', max_ep_sec = 5,
                 controllable_joints=[1,2], control_mode = p.TORQUE_CONTROL, 
                 cost_coeffs = {}, obstacle_params = {}):
        
        super().__init__()
        self.time_step = time_step # The interval between two consecutive actions in seconds
        
        self.max_ep_sec = max_ep_sec # Episode length in seconds (In this implementation, the episode length is fixed)
        
        self.ground_speed = ground_speed # The ground (conveyor belt) linear speed
        
        self.t_res = int(240*time_step) # The number of simulator step calls between two consecutive action querries
                                        # The pybullet simulator was optimized at 240 hz. 
                                        # Therefore, if you want to have 0.1 second progress in simulation, 
                                        # you need to call pybullet step function 24 times.
                    
        self.controllable_joints = controllable_joints # The index of joints, whose motors should be controlled
                                                       # Please see and run "legged.ipynb" for figuring out the joint and link indices, 
                                                       # and to get information regarding your robot.
        
        # Obstacle creation default parameters
        self.obstacle_params = dict(num_obstacles = 2 ,
                                    obst_type = 'box' ,
                                    box_xyz = [0.5,0.2,0.2] ,
                                    random_obstacle_time_dist_maker = lambda : 1. + np.random.uniform(0., 0.3) )
        # Setting the provided obstacle parameters by the user
        for key,val in obstacle_params.items():
            self.obstacle_params[key] = val
        self.num_obstacles = self.obstacle_params # The number of obstacles
        self.obstacle_list = [] # The list of obstacle object unique body ids.
        
        assert self.t_res/self.time_step == 240, \
               'Please provide a time step which is an integer multiple of 1/240 seconds'
        
        self.gui = gui # Whether to use pybullet gui. For training purposes, this should be turned off and set to False.
        if self.gui:
            self.physicsClientId = p.connect(p.GUI)
        else:
            self.physicsClientId = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # Adding "plane.urdf" default path to urdf search path
        self.robot_urdf = robot_urdf # The path of the urdf file
        self.control_mode = control_mode # The control mode of the selected conrollable joints. 
                                         # It's either p.TORQUE_CONTROL, p.VELOCITY_CONTROL, or p.POSITION_CONTROL
        
        self.get_robot_info() # Creating a fake robot, and inquring the necessary information. 
                              # The robot will be destroyed after the getting the necessary info.
                              # This will set some self attributes, such as creating self.sample_init_state.
                              # These attributes will be used later for constructing the gym-related material.
        
        # Advice: Do not be picky about the limits. Pybullet can generate very weird numbers for no good reason.
        self.high_state_limits = [np.finfo(np.float32).max for _ in self.sample_init_state] 
        self.low_state_limits = [-1*np.finfo(np.float32).max for _ in self.sample_init_state]
        self.high_action_limits = []
        self.low_action_limits = []
        
        for i in self.controllable_joints:
            self.high_action_limits += [self.joints_info[i][10]]
            self.low_action_limits += [-1 * self.joints_info[i][10]]
        
        self.high_state_limits = np.array(self.high_state_limits)
        self.low_state_limits = np.array(self.low_state_limits)
        self.high_action_limits = np.array(self.high_action_limits)
        self.low_action_limits = np.array(self.low_action_limits)
        
        # Creating action space
        # We only deal with continuous action spaces in this environment
        self.action_space = spaces.Box(self.low_action_limits, self.high_action_limits, dtype=np.float32)
        
        # The last state, and the taken action can be useful when defining r(s,a,s'). 
        # Therefore, we will prepare them, but one can ignore them.
        self.last_action = self.action_space.sample() #Just a random last state, it's going to get reset in the reset method
        
        #Creating observation space
        self.observation_space = spaces.Box(self.low_state_limits, self.high_state_limits, dtype=np.float32)
        
        # Seeding numpy for initial random state.
        self.seed()
        
        # Objects, like the robot and the ground plane, cannot be instantiated at every reset. 
        # Instantiating requires reading a file from HDD, which could slow down and crash the program.
        # Therefore, we will instantiate them once at the first reset. 
        # In the next reset calls, we will just recover the first state, and reinitialize the "angles, speeds, etc." randomly.
        self.init_state_id = None
        
        # Creating video compiler, and configuring the cameras
        self.configure_dispaly()
        
        # The default cost coefficients
        self.electricity_cost	 = -0.2	# the "per joule price" of motor electricity used. 
                                        # The energy consumption is computed using the 
                                        #   "energy cost = tau * (theta_second - theta_init)"  
                                        # formula after each taken action.
        
        self.stall_torque_cost	= -0.	# cost coefficient for running electric current through a motor even at zero rotational speed. 
                                        # The stall torque is computed this way:
                                        # stall_torque_cost = (self.stall_torque_cost / number of episode steps) * sum(action ^ 2)
        
        self.joints_at_limit_cost = -200.	# cost coefficient for discouraging stuck joints, and going to the limits
                                            # joints_at_limit_cost  = (self.joints_at_limit_cost / number of episode steps) * (number of joints beyond the set thresholds)
        self.joint_penalization_limits = [1.4, 2.8] # The angle thresholds above which a joint will be penalized
        
        #Just a sanity check...
        assert len(self.joint_penalization_limits) == len(self.controllable_joints), \
               'Please modify self.joint_penalization_limits according to your controllable joints choice.'
            
        self.foot_collision_cost  = -300.   # The cost of colliding with either yourself (for instance, upper leg hitting the base), 
                                            # or collision of a non-lower-leg (i.e. the base box or the upper leg) link with ground plane.
                                            # foot_collision_cost = (self.foot_collision_cost / number of episode steps) * The number of "bad" collisions described above
        
        self.obstacle_collision_cost = -2000. # The cost of colliding with an obstacle
                                              # obstacle_collision_cost = (self.obstacle_collision_cost / number of episode steps) * The number of obstacle collisions
        
        self.speed_cost = -0. # Penalizing too much angular velocity. 
                              # speed_cost = (self.speed_cost / number of episode steps) * np.sum(np.abs(joint angular velocities vector))
            
        self.max_height_cost = -200. # The cost for jumping too high
                                     # max_height_cost = (self.max_height_cost / number of episode steps) * (1 if potential energy is above a threshold, and otherwise 0).
        
        # Setting the provided cost coefficients by the user
        for key,val in cost_coeffs.items():
            setattr(self, key, val)
            
    def seed(self, seed=None):
        #Seeding as done usually by gym environments.
        (self.np_random, seed) = seeding.np_random(seed)
        return [seed]
    
    def reset(self, init_state_vec = None):
        # init_state_vec could be used for providing a specific initial angles and speeds
        # For instance:
        # init_state_vec = [1., 0.1, 2., 0.3] means that 
        # theta_1 = 1., omega_1 = 0.1, theta_2 = 2., omega_2 = 0.3
        
        #Reseting the number of step function calls count
        self.curr_num_steps = 0
        
        #Getting obstacle params
        obst_type = self.obstacle_params['obst_type']
        box_xyz = self.obstacle_params['box_xyz']
        random_obstacle_time_dist_maker = self.obstacle_params['random_obstacle_time_dist_maker']
        
        # Checking if we need to load the urdfs from file, or use previously loaded objects
        if self.init_state_id is None:
            # The objects were not instantiated before. Therefore, we will need to instantiate them.
            
            # Scaling the ground plane, so that we don't run out of conveyor belt!
            ground_scaling = max(1., 1.3 * self.ground_speed * self.max_ep_sec / 15.)
            p.resetSimulation(physicsClientId = self.physicsClientId)
            self.plane = p.loadURDF("plane.urdf", useFixedBase=True, globalScaling = ground_scaling, 
                                    physicsClientId = self.physicsClientId)
            p.resetBaseVelocity(objectUniqueId=self.plane, linearVelocity=[0.,self.ground_speed,0.], 
                                physicsClientId = self.physicsClientId)
            
            # robot instantiation
            self.robot = p.loadURDF(self.robot_urdf, basePosition = [0, 0, 1.], 
                                    baseOrientation = [0,0,1,1], useFixedBase=False,
                                    flags = p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT + p.URDF_USE_SELF_COLLISION,
                                    physicsClientId = self.physicsClientId)
            
            # Creating the obstacles
            for pp in range(self.num_obstacles):
                self.create_obstacle(obst_type='box', box_xyz = box_xyz, mass = obstacle_mass, 
                                     baseTimeDist = (pp + 1.))
            
            # Changing the friction coefficeints of robot links and the ground plane.
            # If you think this is necessary, turn it on. It's disabled by default.
            change_friction_coeffs = False
            if change_friction_coeffs:
                for bodyunqid in [self.plane, self.robot]:
                    self.change_friction_coeffs(bodyUniqueId = bodyunqid, physicsClientId = self.physicsClientId, 
                                                links_idx_list='all', lateral_coeff_list = 1., spinning_friction_list = 1.,
                                                rolling_friction_list = 1.)
            
            self.get_robot_info() #Just to make sure that everything is updated
            
            # Saving the world state, with the objects to be restored in later reset functions.
            self.init_state_id = p.saveState(physicsClientId = self.physicsClientId)
            
        else:
            # Restoring the formerly created objects.
            p.restoreState(stateId = self.init_state_id, physicsClientId = self.physicsClientId)
        
        
        # Putting the obstacles in random distances
        for yy, obstacle in enumerate(self.obstacle_list):
            baseTimeDist = random_obstacle_time_dist_maker() * (yy+1.) 
            posObj = [0, -0.35 - box_xyz[1]/2 * (2.*yy + 1.) - self.ground_speed * baseTimeDist, box_xyz[2]/2 ]
            p.resetBasePositionAndOrientation(obstacle, posObj = posObj, ornObj=[0.,0.,0.,1.], physicsClientId = self.physicsClientId)
        
        # A sanity check on whether we're controlling the right joints.
        for uu in self.controllable_joints:
            assert self.joints_info[uu][1].decode('ASCII') in ['base_to_upperleg', 'upperleg_to_lowerleg'], self.joints_info[uu][-5].decode('ASCII')
        
        # Visualization utilties
        self.viz_util = pyb_viz(physicsClientId = self.physicsClientId)
        
        # Deactivating all the joint motors! This is super necessary for proper torque control.
        for jointIndex in range(p.getNumJoints(self.robot, physicsClientId = self.physicsClientId)):
            maxForce = 0
            mode = p.VELOCITY_CONTROL
            p.setJointMotorControl2(self.robot, jointIndex, controlMode=mode, force=maxForce, physicsClientId = self.physicsClientId)
        
        # Real Time Simulation should be disabled. Otherwise, the simulation will just go on for itself without even calling the step function.
        p.setRealTimeSimulation(0, physicsClientId = self.physicsClientId)
        
        # Setting Gravity
        p.setGravity(0, 0, -9.81, physicsClientId = self.physicsClientId)
        
        #Setting random 
        for pp,joint_id in enumerate(self.controllable_joints):
            if init_state_vec is None:
                rand_pos = self.np_random.uniform(low=-0.5, high=0.5)
                rand_vel = 0.0
                p.resetJointState(self.robot, joint_id, rand_pos, rand_vel,
                                  physicsClientId = self.physicsClientId)
            else:
                p.resetJointState(self.robot, joint_id, init_state_vec[2*pp], 
                                  init_state_vec[2*pp+1],
                                  physicsClientId = self.physicsClientId)
        
        # Setting the inital state material
        self.state, self.state_dict, self.potential = self.get_state(robot = self.robot, physicsClientId = self.physicsClientId)
        
        # Resetting the video compiler and the cameras.
        self.configure_dispaly()

        return np.array(self.state)
    
    def step(self, action):
        # action could be a list or an array.
        
        # Calling the stepSimulation function of pybullet.
        # Since I don't trust pybullet, I set the joint control commands before every stepsim call!
        for _ in range(self.t_res):
            for i,joint_idx in enumerate(self.controllable_joints):
                if self.control_mode == p.TORQUE_CONTROL:
                    #p.setJointMotorControl2(self.robot, joint_idx, p.VELOCITY_CONTROL, force=0, physicsClientId = self.physicsClientId)
                    p.setJointMotorControl2(self.robot, joint_idx, p.TORQUE_CONTROL, force=action[i], physicsClientId = self.physicsClientId)
                elif self.control_mode == p.VELOCITY_CONTROL:
                    p.setJointMotorControl2(self.robot, joint_idx, p.VELOCITY_CONTROL, targetVelocity=action[i],
                                            force = 10,
                                            physicsClientId = self.physicsClientId)
                elif self.control_mode == p.POSITION_CONTROL:
                    p.setJointMotorControl2(self.robot, joint_idx, p.POSITION_CONTROL, targetPosition=action[i], physicsClientId = self.physicsClientId)
                else:
                    raise('Unknown Control Mode')
            p.stepSimulation(physicsClientId = self.physicsClientId)
        
        # Remembering the last taken action. This could be useful when computing the reward r(s,a,s').
        self.last_action = action
        # Keeping the old state vector, and relate material. This could be useful when computing the reward r(s,a,s').
        self.last_state, self.last_state_dict, self.last_potential = self.state, self.state_dict, self.potential
        # Computing the new state material.
        self.state, self.state_dict, self.potential = self.get_state(robot = self.robot, physicsClientId = self.physicsClientId)
        
        # Computing reward and done
        reward, done = self.get_reward_and_done()
        
        # For better access, I report the state dict as the info output.
        info = self.state_dict
        
        # The number of gym environment step function calls since the last reset.
        self.curr_num_steps += 1

        return (np.array(self.state), reward, done, info)
        
    def get_state(self, robot, physicsClientId):
        # This method needs the robot unique body id, and also the physics client id you have connected to
        # This method could be applied on the fake robot created for information extraction purposes
        # Please know that this function could be called before even the ground plane and other objects are instantiated.
        
        #Joint States
        all_joints = range(p.getNumJoints(robot, physicsClientId = physicsClientId))
        joint_states = [list(x) for x in p.getJointStates(robot, all_joints, physicsClientId = physicsClientId)]
        joint_angles = [x[0] for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]
        joint_torques = [x[3] for x in joint_states]
        
        #Link States
        link_states=[]
        for link_id in all_joints:
            link_states.append(p.getLinkState(robot, linkIndex=link_id, computeLinkVelocity=True, physicsClientId = physicsClientId))
        link_world_pos = np.array([x[0] for x in link_states]).tolist()
        link_orientations = np.array([p.getEulerFromQuaternion(x[1]) for x in link_states]).reshape(-1).tolist()
        link_lin_vel = np.array([x[6][1:] for x in link_states]).reshape(-1).tolist() #Only along y,z axis, since the policy should be independent of robot x coordinates.
        link_ang_vel = np.array([x[7][1:] for x in link_states]).reshape(-1).tolist() #Only along y,z axis
        
        #Base State
        base_position, base_orientation = p.getBasePositionAndOrientation(robot, physicsClientId = physicsClientId)
        base_z = [base_position[2]]
        base_orientation = np.array(p.getEulerFromQuaternion(base_orientation)).reshape(-1).tolist()
        
        # Contact Information processing
        # Pybullet returns a list of contacts. Two links can have contacts in more than one point
        # Since we are only interested in whether a contact happend or not, we would remove duplicate contact points of same pairs of links.
        lower_leg_idxs =[2]
        contc_pts_info = p.getContactPoints(physicsClientId = physicsClientId)
        unq_contc_pts_info = {}
        for contct in contc_pts_info:
            if (contct[1], contct[2], contct[3], contct[4]) in unq_contc_pts_info.keys():
                continue
            if (contct[2], contct[1], contct[4], contct[2]) in unq_contc_pts_info.keys():
                continue
            unq_contc_pts_info[(contct[1], contct[2], contct[3], contct[4])]=contct
        contc_pts_info=list(unq_contc_pts_info.values())
        
        #Profiling the type of contacts
        good_contcts = 0. #lower leg to ground contact
        bad_contcts = 0. #robot link to robot link contcts and non-lower-leg to ground contacts
        obstacle_contcts = 0.
        nonrobot_contcts = 0.
        
        # Since this method could be called even before the ground plane is created, we will have to make sure that the function would not crash.
        # The calls before creating the ground plane are not needed to be accurate. They just need to provide the same state dimensions, as if the sim was running.
        # In case plane ground was not defined, we will use any secondary object as the ground plane so that the function would not crash!
        if hasattr(self, 'plane'):
            safe_hit_obj = self.plane
        else:
            safe_hit_obj = robot
            
        #Classifying the contacts
        for contct in contc_pts_info:
            a = contct[1]
            b = contct[2]
            a_link = contct[3]
            b_link = contct[4]
            
            if a == b == robot:
                # robot link to robot link collision. This is a bad contact, and would be used for 
                # penalization later in get_reward_and_done.
                # Note: bad_contacts and obstacle_contacts are stored in self.state_dict, 
                #       and get_reward_and_done will have access to them later using self.state_dict.
                bad_contcts = bad_contcts + 1
                
            elif (a == robot) or (b == robot):
                # robot to non-robot collision.
                
                # recognizing the robot from the other colliding object
                robot_link = a_link if a == robot else b_link
                contct_obj = b if a == robot else a
                contct_obj_link = b_link if a == robot else a_link
                
                # collision classification
                if contct_obj == safe_hit_obj:
                    if robot_link in lower_leg_idxs:
                        # Lower robot leg to ground plane hit is a good contact, and should not be penalized.
                        good_contcts = good_contcts + 1
                    else:
                        # Upper leg or base box contact to the ground plane is a bad contact, and should be penalized.
                        bad_contcts = bad_contcts + 1
                else:
                    # If one collidin object is the robot, and the other is not the ground plane, then it's an obstacle contact.
                    # We may need to penalize the obstacle contact differently, and this is why we make a distinction between bad and obstacle contacts.
                    obstacle_contcts = obstacle_contcts + 1                    
            else:
                # If neither of colliding objects is the robot, then it's irrelevant.
                nonrobot_contcts = nonrobot_contcts + 1
        
        # We will include the links and base position in the state vector, along with the orientations.
        link_poses = np.array(link_world_pos).reshape(-1).tolist()
        link_poses += np.array(base_position).reshape(-1).tolist()
        link_ornts = link_orientations + base_orientation
        
        # Obstacles
        obstacle_poses = []   
        obstacle_ornts = []
        obstacle_list = self.obstacle_list[:] # Creating a deep copy of the obstacle unique ids
        
        # Making sure that the obstacles are there. In case they were not created, we will use any object such as the robot itself!
        while len(obstacle_list) < self.num_obstacles:
            obstacle_list = obstacle_list + [robot] #Just for the get_robot_info method, where the obstacles have not been created yet!
        
        # inquiring about obstacle position and orientations to be included in the state vector
        for obstacle in obstacle_list:
            obstacle_pos, obstacle_ornt_q = p.getBasePositionAndOrientation(obstacle, physicsClientId = physicsClientId)
            obstacle_ornt = p.getEulerFromQuaternion(obstacle_ornt_q)
            obstacle_poses.append(obstacle_pos)
            obstacle_ornts.append(obstacle_ornt)
            
        obstacle_poses = np.array(obstacle_poses).reshape(-1).tolist()
        obstacle_ornts = np.array(obstacle_ornts).reshape(-1).tolist()
        
        # Compiling different state components. Each component is a list, and so adding them will give us the concatenation of them.
        state = joint_angles + joint_velocities + link_poses +  link_ornts + link_lin_vel + link_ang_vel + obstacle_poses + obstacle_ornts
        
        # State dictionary is a dictionary, that could be used for carrying information which you don't want to show to the robot.
        # These extra pieces of information, could be useful for computing reward, or printing them for your own purposes.
        state_dict = dict(joint_angles= joint_angles, joint_velocities = joint_velocities,
                          link_lin_vel = link_lin_vel,  link_ang_vel=link_ang_vel, link_orientations = link_orientations,
                          base_z = base_z, base_orientation = base_orientation, link_world_pos = link_world_pos, bad_contcts = bad_contcts, 
                          obstacle_contcts = obstacle_contcts, nonrobot_contcts = nonrobot_contcts,
                          good_contcts = good_contcts, joint_torques = joint_torques)
        
        
        # Creating a string of [theta, omega, action, applied force] lists for later prinitn on renderred images.
        joint_state_np = np.array([[x[0], x[1], x[3]] for i,x in enumerate(joint_states) if i in self.controllable_joints])
        debug_list = joint_state_np.reshape(-1).tolist()
        state_dict['s_sdot_a'] = debug_list
        if hasattr(self,'last_action'):
            info_np = np.array([[x[0], x[1], self.last_action[self.controllable_joints.index(i)], x[3]] 
                                for i,x in enumerate(joint_states) if i in self.controllable_joints])
            state_dict['s_sdot_a_f'] = info_np
        
        # Computing the potential energy
        potential = self.get_potential(state_dict = state_dict)
        
        return state, state_dict, potential
    
    def get_potential(self, state_dict = None):
        # Computing the potential energy of the robot.
        if state_dict is None:
            state_dict = self.state_dict
        link_world_pos = state_dict['link_world_pos']
        link_z = [pos[2] for pos in link_world_pos]
        link_z += state_dict['base_z'] #Last one is the height of base
        link_z = np.array(link_z)
        potential = np.sum(np.array(self.link_masses) * link_z) * 9.81
        return potential
    
    def get_reward_and_done(self):
        # This function computes the reward, and whether the episode should be finished.
        # State material comes from the self.state_dict dictionary.
        joint_angles = self.state_dict['joint_angles']
        last_joint_angles = self.last_state_dict['joint_angles']
        joint_velocities = self.state_dict['joint_velocities']
        joint_torques = self.state_dict['joint_torques']
        
        # a is the motor torques. in velocity control mode, they should be asked from the environment.
        if self.control_mode == p.TORQUE_CONTROL:
            a = np.array(self.last_action)
        else:
            a = np.array([joint_torques[i] for i in self.controllable_joints])
        
        # controlled joint angle, position, velocity, etc. information.
        cont_theta = np.array([joint_angles[i] for i in self.controllable_joints])
        last_cont_theta = np.array([last_joint_angles[i] for i in self.controllable_joints])
        cont_omega = np.array([joint_velocities[i] for i in self.controllable_joints])
        
        
        # Computing the costs. For more information on the cost coefficients and how they work, see the constructor.
        
        # The electricity consumption costs
        electricity_cost  = self.electricity_cost  * np.sum(a*(cont_theta - last_cont_theta))
        electricity_cost += self.stall_torque_cost * np.sum(a**2) * self.time_step / self.max_ep_sec
        
        # Speed cost
        speed_cost = self.speed_cost * np.sum(np.abs(cont_omega)) * self.time_step / self.max_ep_sec
        
        # Joint at limits cost
        joints_at_limit = np.maximum(0., np.abs(cont_theta)-np.array(self.joint_penalization_limits))
        joints_at_limit_cost = self.joints_at_limit_cost * np.sum(joints_at_limit) * self.time_step / self.max_ep_sec
        
        # Potential Energy progress
        progress = self.potential - self.last_potential
        
        # Penalty for jumping too high
        if self.potential > 30:
            # Hard penalty for jumping above almost 3 meters.
            progress = -1 * progress + self.max_height_cost * self.time_step / self.max_ep_sec
        elif self.potential > 15:
            # Soft penalty for jumping above almost 1.5 meters.
            progress = -1 * progress
        
        # Penalty for having collisions. Bad contacts are distinct from obstacle contacts, since they should be rewarded differently.
        bad_contcts = self.state_dict['bad_contcts']
        obstacle_contcts = self.state_dict['obstacle_contcts']
        collision_cost = bad_contcts * self.foot_collision_cost * self.time_step / self.max_ep_sec
        collision_cost = collision_cost + obstacle_contcts * self.obstacle_collision_cost * self.time_step / self.max_ep_sec
        
        # Computing the final reward.
        reward = joints_at_limit_cost + electricity_cost + progress + collision_cost + speed_cost
        
        # Computing whether the sim should be reset
        done = (self.curr_num_steps * self.time_step) > self.max_ep_sec
        
        return reward, done
    
    def create_obstacle(self, obst_type='box', box_xyz=[1,0.2,0.3], mass=1, baseTimeDist=1.0):
        # Creating an obstacle and appending it to the self.obstacle_list
        # box_xyz: The dimensions of the box, in case you want to create a box!
        # mass: Mass of the obstacle.
        # baseTimeDist: The time distance to the robot, considering the ground speed.
        if obst_type=='box':
            half_extent = (np.array(box_xyz)/2).tolist()
            basePosition = [0, -1 * baseTimeDist * self.ground_speed, half_extent[2]]
            colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents = half_extent, physicsClientId = self.physicsClientId)
            vizShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents = half_extent, rgbaColor=[1,0,0,1], physicsClientId = self.physicsClientId)
            obstacleUid = p.createMultiBody(baseMass = mass, baseCollisionShapeIndex = colShapeId, baseVisualShapeIndex = vizShapeId, 
                                          basePosition = basePosition, baseOrientation = [0,0,0,1], physicsClientId = self.physicsClientId)
            self.obstacle_list.append(obstacleUid)
            return obstacleUid
        else:
            raise 'Uknown obst_type'
    
        
    def change_friction_coeffs(self, bodyUniqueId, physicsClientId, 
                               links_idx_list='all', 
                               lateral_coeff_list = 1., spinning_friction_list = 0.,
                               rolling_friction_list = 0.):
        
        # Changing the friction coeffs of pybullet after loading the urdf.
        # Note: This is not called by default. Modify the reset function if you would like to apply this function
        if isinstance(links_idx_list, str):
            if links_idx_list.lower() == 'all':
                links_idx_list = list(range(p.getNumJoints(bodyUniqueId, physicsClientId = physicsClientId))) + [-1]
        
        def convert_to_list(x):
            if not isinstance(x, collections.Iterable):
                y = [x for _ in links_idx_list]
            return y
        lateral_coeff_list = convert_to_list(lateral_coeff_list)
        spinning_friction_list = convert_to_list(spinning_friction_list)
        rolling_friction_list = convert_to_list(rolling_friction_list)
            
        iter_material = zip(links_idx_list, rolling_friction_list, lateral_coeff_list, spinning_friction_list)
        for linkIndex, rollingFriction, lateralFriction, spinningFriction in iter_material:
            p.changeDynamics(bodyUniqueId = bodyUniqueId, linkIndex = linkIndex, 
                             lateralFriction = lateralFriction, rollingFriction = rollingFriction,
                             spinningFriction = spinningFriction, physicsClientId = physicsClientId)
    
    def get_robot_info(self):
        #We will run a fake connection if the robot does not exist yet.
        #Just load the urdf, and inquire about the robot information
        #Then we will disconnect, and kill the robot
        if hasattr(self, 'robot') and hasattr(self, 'physicsClientId'):
            #The robot exists, and therefore no need to make a fake robot
            fake_robot_creation = False
            fakeclient = self.physicsClientId
            fakerobot = self.robot
        else:
            #The fake robot should be created.
            fake_robot_creation = True
            fh = open(self.robot_urdf, "r")
            name_line = fh.readlines()[0]
            fh.close()
            assert '<robot name' in name_line, 'The first line in URDF should have the name of the robot. ' \
                                                + 'Please transfor the "<robot name" line to be the first line of the urdf file.'
            robot_name = name_line.split('"')[1]
            self.robot_name = robot_name
            
            fakeclient = p.connect(p.DIRECT)
            fakerobot = p.loadURDF(self.robot_urdf, physicsClientId = fakeclient)
            
        self.joints_info = []
        self.links_info = []
        # All joint indices (i.e. controllable or non-controllable)
        self.all_joints = list(range(p.getNumJoints(fakerobot, physicsClientId = fakeclient)))
        for i in self.all_joints:
            self.joints_info.append(list(p.getJointInfo(fakerobot, jointIndex = i, physicsClientId = fakeclient)))
            self.links_info.append(list(p.getDynamicsInfo(fakerobot, linkIndex = i, physicsClientId = fakeclient)))
        self.links_info.append(list(p.getDynamicsInfo(fakerobot, linkIndex = -1, physicsClientId = fakeclient)))
        self.link_masses = [self.links_info[i][0] for i in self.all_joints + [-1]]
        
        self.sample_init_state, self.sample_init_state_dict, self.sample_init_potential = self.get_state(fakerobot, fakeclient)
        if fake_robot_creation:
            p.disconnect(physicsClientId = fakeclient)
            
        #Setting the joint limits according to urdf file
        self.low_joint_limits = [self.joints_info[joint_idx][8] for joint_idx in self.all_joints]
        self.high_joint_limits = [self.joints_info[joint_idx][9] for joint_idx in self.all_joints]
        
        return self.joints_info, self.sample_init_state
        
    
    def print_joints_info(self):
        #Printing robot information in a nice format using pandas
        col_names = ['Index', 'Name', 'Type', 'qIndex', 'uIndex', 'flags', 'Damping', 'Friction', 'Lower Limit', 
                     'Upper Limit', 'Max Force', 'Max Velocity', 'Link Name', 'Axis', 'parentFramePos', 'parentFrameOrn', 'parentIndex']
        link_col_names = ['Mass', 'Lateral friction', 'Local inertia diagonal', 'Local inertial pos', 'Local inertial orn', 
                             'Restitution', 'Rolling friction', 'Spinning friction', 'Contact damping', 'Contact stiffness']
        link_col_names = ['Link Index','Link name'] + link_col_names
        
        type_translation = {p.JOINT_REVOLUTE: 'Revolute', p.JOINT_PRISMATIC:'Prismatic',
                            p.JOINT_SPHERICAL: 'Spherical', p.JOINT_PLANAR:'Planar', p.JOINT_FIXED: 'Fixed'}
        
        import copy
        joints_info = copy.deepcopy(self.joints_info)
        links_info = copy.deepcopy(self.links_info)
        for i in range(len(joints_info)):
            joints_info[i][2] = type_translation[joints_info[i][2]]
            links_info[i] = [joints_info[i][0],joints_info[i][-5]] + links_info[i]
        links_info[-1] = ['-1', 'Base'] + links_info[-1]

        try:
            import pandas as pd
            df = pd.DataFrame(joints_info, columns = col_names)
            print('Joints Information Table: ')
            print(df[['Index', 'Name', 'Type', 'Damping', 'Friction', 'Lower Limit', 
                     'Upper Limit', 'Max Force', 'Max Velocity', 'Link Name', 'Axis']])
            print('--------------------------------------------')
            print('\n')
            print('Links Information Table: ')
            df = pd.DataFrame(links_info, columns = link_col_names)
            print(df[['Link name', 'Link Index', 'Lateral friction', 'Rolling friction', 
                      'Spinning friction', 'Contact damping', 'Contact stiffness', 'Restitution']])
            
        except ImportError:
            print(col_names)
            for i in range(len(joints_info)):
                print(joints_info[i])
        
        print('\nJoints ' + str([joints_info[joint_idx][1].decode("utf-8") for joint_idx in self.controllable_joints]) + ' will be controlled during the simulation.')
            

    def configure_dispaly(self, pixelWidth=640, pixelHeight=480):
        # Configuring the video compiler, which takes all the rgb images created by the render function, and compiles them in itself.
        # This would be useful for recording vidoes.
        self.vid_compiler = compiler_video()
        
        # Also, we need to set the camera parameters
        self.im_renderer_h = image_renderer(pixelWidth=pixelWidth, pixelHeight=pixelHeight, physicsClientId = self.physicsClientId)
        self.im_renderer_v = image_renderer(pixelWidth=pixelWidth, pixelHeight=pixelHeight, physicsClientId = self.physicsClientId)
        self.im_renderer_h.set_view(yaw = 90, #yaw angle in degrees left/right around up-axis
                                    pitch = -20, #pitch in degrees up/down
                                    roll = 0) #roll in degrees around forward vector
        self.im_renderer_v.set_view(yaw = 0, pitch=-20, roll=0)
        
    def render(self, mode='human', close=False):
        if self.gui:
            return 
        np_img_arr_h = self.im_renderer_h()
        np_img_arr_v = self.im_renderer_v()
        border = np.zeros((np_img_arr_h.shape[0], 10, 4))
        np_img_arr = np.concatenate([np_img_arr_v, border, np_img_arr_h], axis=1)
        
        
        pil_im = Image.fromarray(np_img_arr.astype(np.uint8))
        d = ImageDraw.Draw(pil_im)
        im_string = 's, w, a, f: \n' + np.array2string(np.array(self.state_dict['s_sdot_a_f']), precision=2, 
                                                       separator=',',suppress_small=True) 
        im_string = im_string + '\n [good, bad, obstacle]: ' + np.array2string(np.array([self.state_dict['good_contcts'], self.state_dict['bad_contcts'], 
                                                                                         self.state_dict['obstacle_contcts']]), precision=2)
        if hasattr(self,'debug_str'):
            im_string = im_string + '\n' + self.debug_str
        d.text((10,10), im_string, fill=(0,0,0))
        np_img_arr = np.array(pil_im)
        
        self.vid_compiler.add_np_img(np_img_arr)
        return np_img_arr
    
    def compile_video(self, out_file='test.mp4', fps = 20):
        clip = self.vid_compiler(out_file=out_file, fps=fps)
        print('done video compiling!')
        return clip

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def get_angle(cosine, sine):
    possine = 2*(np.array(sine) > 0) - 1
    theta = possine * np.arccos(cosine)
    theta = np.mod(theta, 2*np.pi)
    return theta