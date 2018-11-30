import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p
import pybullet_data
from utilities.render_utils import compiler_video, image_renderer
from utilities.pyb_vizutils import pyb_viz
from utilities.pyb_bodyutils import create_side_sliders, deactivate_joint_motors

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class robotBulletEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 50}

    def __init__(self, gui = True, time_step = 0.0001, t_res = 1,
                 robot_urdf = 'urdf/mymonoped_handed.urdf',
                 controllable_joints=[1,2], control_mode = p.TORQUE_CONTROL):
        
        self.time_step = time_step
        self.t_res = t_res
        assert self.t_res/self.time_step == 240
        self.gui = gui
        if self.gui:
            self.physicsClientId = p.connect(p.GUI)
        else:
            self.physicsClientId = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_urdf = robot_urdf
        self.control_mode = control_mode
        self.get_robot_info()
        assert self.robot_name in ['monoped', 'updown_box', 'pendulum']
        
        self.controllable_joints = controllable_joints
        
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
        
        self.discrete_action = False
        if self.discrete_action:
            #TODO: somehow manage the torque control in discrete case
            assert not(self.discrete_action), 'Not implemented yet!'
            self.force_mag = 10
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(self.low_action_limits, self.high_action_limits, dtype=np.float32)
            
        self.observation_space = spaces.Box(self.low_state_limits, self.high_state_limits, dtype=np.float32)
        
        self.seed()
        self.init_state_id = None
        self.configure_dispaly()
        
        self.electricity_cost	 = -2.0  * 0.0	# cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
        self.stall_torque_cost	= -0.1   * 0.0 	# cost for running electric current through a motor even at zero rotational speed, small
        self.joints_at_limit_cost = -0.1 * 0.0 	# discourage stuck joints
        self.foot_collision_cost  = -1.0 * 0.0
    
    def get_robot_info(self):
        #We will run a fake connection
        #Just load the urdf, and inquire about the robot information
        #Then we will disconnect, and kill the robot
        if hasattr(self, 'robot') and hasattr(self, 'physicsClientId'):
            fake_robot_creation = False
            fakeclient = self.physicsClientId
            fakerobot = self.robot
        else:
            fake_robot_creation = True
            
            fh = open(self.robot_urdf, "r")
            name_line = fh.readlines()[0]
            fh.close()
            assert '<robot name' in name_line, 'The first line in URDF should have the name of the robot'
            robot_name = name_line.split('"')[1]
            self.robot_name = robot_name
            
            fakeclient = p.connect(p.DIRECT)
            fakerobot = p.loadURDF(self.robot_urdf, physicsClientId = fakeclient)
        self.joints_info = []
        self.links_info = []
        self.all_joints = list(range(p.getNumJoints(fakerobot, physicsClientId = fakeclient)))
        for i in self.all_joints:
            self.joints_info.append(list(p.getJointInfo(fakerobot, jointIndex = i, physicsClientId = fakeclient)))
            self.links_info.append(list(p.getDynamicsInfo(fakerobot, linkIndex = i, physicsClientId = fakeclient)))
        self.links_info.append(list(p.getDynamicsInfo(fakerobot, linkIndex = -1, physicsClientId = fakeclient)))
        self.link_masses = [self.links_info[i][0] for i in self.all_joints + [-1]]
        
        self.sample_init_state, self.sample_init_state_dict, self.sample_init_potential = self.get_state(fakerobot, fakeclient)
        if fake_robot_creation:
            p.disconnect(physicsClientId = fakeclient)
        self.low_joint_limits = [self.joints_info[joint_idx][8] for joint_idx in self.all_joints]
        self.high_joint_limits = [self.joints_info[joint_idx][9] for joint_idx in self.all_joints]
        return self.joints_info, self.sample_init_state
        
    
    def print_joints_info(self):
        
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
        self.vid_compiler = compiler_video()
        self.im_renderer_h = image_renderer(pixelWidth=pixelWidth, pixelHeight=pixelHeight, physicsClientId = self.physicsClientId)
        self.im_renderer_v = image_renderer(pixelWidth=pixelWidth, pixelHeight=pixelHeight, physicsClientId = self.physicsClientId)
        self.im_renderer_h.set_view(yaw = 90, #yaw angle in degrees left/right around up-axis
                                    pitch = -20, #pitch in degrees up/down
                                    roll = 0) #roll in degrees around forward vector
        self.im_renderer_v.set_view(yaw = 0, pitch=-20, roll=0)

    def seed(self, seed=None):
        (self.np_random, seed) = seeding.np_random(seed)
        return [seed]
    
    def get_state(self, robot, physicsClientId):
        #Joint States
        all_joints = range(p.getNumJoints(robot, physicsClientId = physicsClientId))
        joint_states = [list(x) for x in p.getJointStates(robot, all_joints, physicsClientId = physicsClientId)]
        joint_angles = [x[0] for x in joint_states]
        if self.robot_name == 'pendulum':
            joint_angles = [np.cos(x[0]) for x in joint_states] + [np.sin(x[0]) for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]
        joint_torques = [x[3] for x in joint_states]
        
        #Link States
        link_states=[]
        for link_id in all_joints:
            link_states.append(p.getLinkState(robot, linkIndex=link_id, computeLinkVelocity=True, physicsClientId = physicsClientId))
            
        link_world_pos = np.array([x[0] for x in link_states]).tolist()
        link_orientations = np.array([p.getEulerFromQuaternion(x[1]) for x in link_states]).reshape(-1).tolist()
        link_lin_vel = np.array([x[6][1:] for x in link_states]).reshape(-1).tolist() #Only along y,z axis
        link_ang_vel = np.array([x[7][1:] for x in link_states]).reshape(-1).tolist() #Only along y,z axis
        
        #Base State
        base_position, base_orientation = p.getBasePositionAndOrientation(robot, physicsClientId = physicsClientId)
        base_z = [base_position[2]]
        base_orientation = np.array(p.getEulerFromQuaternion(base_orientation)).reshape(-1).tolist()
        
        #Contact Info
        lower_leg_idxs =[4,0]# 4 was the lower leg link at the time I wrote this, and 0 was the slider link
        contc_pts_info = p.getContactPoints(physicsClientId = self.physicsClientId)
        good_contcts = 0.
        for contct in contc_pts_info:
            a = contct[1]
            b = contct[2]
            a_link = contct[3]
            b_link = contct[4]
            for x, x_link in [(a,a_link), (b,b_link)]:
                if x == self.robot and (x_link in lower_leg_idxs):
                    good_contcts += 1
        bad_contcts = len(contc_pts_info) - good_contcts
        
        link_poses = np.array(link_world_pos).reshape(-1).tolist()
        link_poses += np.array(base_position).reshape(-1).tolist()
        link_ornts = link_orientations + base_orientation
        
        state = joint_angles + joint_velocities + joint_torques + link_poses +  link_ornts + link_lin_vel + link_ang_vel 
        
        state_dict = dict(joint_angles= joint_angles, joint_velocities = joint_velocities,
                          link_lin_vel = link_lin_vel,  link_ang_vel=link_ang_vel, link_orientations = link_orientations,
                          base_z = base_z, base_orientation = base_orientation, link_world_pos = link_world_pos, bad_contcts = bad_contcts, 
                          good_contcts = good_contcts)
        
        potential = self.get_potential(state_dict = state_dict)
        return state, state_dict, potential
        
        

    def step(self, action):
        if self.discrete_action:
            #TODO: somehow manage the torque control in discrete case
            assert not(self.discrete_action), 'Not implemented yet!'
        
        for _ in range(self.t_res):
            for i,joint_idx in enumerate(self.controllable_joints):
                if self.control_mode == p.TORQUE_CONTROL:
                    p.setJointMotorControl2(self.robot, joint_idx, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId = self.physicsClientId)
                    p.setJointMotorControl2(self.robot, joint_idx, p.TORQUE_CONTROL, force=action[i], physicsClientId = self.physicsClientId)
                elif self.control_mode == p.VELOCITY_CONTROL:
                    p.setJointMotorControl2(self.robot, joint_idx, p.VELOCITY_CONTROL, targetVelocity=action[i], physicsClientId = self.physicsClientId)
                elif self.control_mode == p.POSITION_CONTROL:
                    p.setJointMotorControl2(self.robot, joint_idx, p.POSITION_CONTROL, targetPosition=action[i], physicsClientId = self.physicsClientId)
                else:
                    raise('Unknown Control Mode')
            p.stepSimulation(physicsClientId = self.physicsClientId)
        
        self.last_action = action
        self.last_state, self.last_state_dict, self.last_potential = self.state, self.state_dict, self.potential
        self.state, self.state_dict, self.potential = self.get_state(robot = self.robot, physicsClientId = self.physicsClientId)
        
        reward, done = self.get_reward_and_done()

        return (np.array(self.state), reward, done, {})
    
    def get_potential(self, state_dict = None):
        if state_dict is None:
            state_dict = self.state_dict
        link_world_pos = state_dict['link_world_pos']
        link_z = [pos[2] for pos in link_world_pos]
        link_z += state_dict['base_z'] #Last one is the height of base
        link_z = np.array(link_z)
        potential = np.sum(np.array(self.link_masses) * link_z)
        return potential
    
    def get_reward_and_done(self):
        
        #state_dict = dict(joint_angles= joint_angles, joint_velocities = joint_velocities,
        #                  link_lin_vel = link_lin_vel,  link_ang_vel=link_ang_vel, link_orientations = link_orientations,
        #                  base_z = base_z, base_orientation = base_orientation)
        
        #z = self.state_dict['base_z'][0]
        if self.robot_name == 'monoped':
            joint_angles = self.state_dict['joint_angles']
            joint_velocities = self.state_dict['joint_velocities']

            a = np.array(self.last_action)
            cont_omega = np.array([joint_angles[i] for i in self.controllable_joints])
            cont_theta = np.array([joint_velocities[i] for i in self.controllable_joints])
            electricity_cost  = self.electricity_cost  * np.mean(np.abs(a*cont_omega))  # let's assume we have DC motor with controller, and reverse current braking
            electricity_cost += self.stall_torque_cost * np.square(a).mean()
            joints_at_limit = [float(not((self.low_joint_limits[i]+0.6) < joint_angles[i] < (self.high_joint_limits[i]-0.6))) for i in self.all_joints]
            joints_at_limit = np.array(joints_at_limit)
            joints_at_limit_cost = np.mean(self.joints_at_limit_cost * joints_at_limit)
            progress = (self.potential - self.last_potential) * 10.0 / self.time_step
            #progress = self.potential

            bad_contcts = self.state_dict['bad_contcts']
            collision_cost = bad_contcts * self.foot_collision_cost

            reward = joints_at_limit_cost + electricity_cost + progress + collision_cost
            #reward = np.abs(self.state[0])
            #reward = -1 * np.mean(np.abs(np.array(self.state) - np.array(self.sample_init_state)))
            done = False
            return reward, done

        elif self.robot_name == 'updown_box':
            joint_z = self.state_dict['joint_angles'][0] #it's not really joint angles. for the updown box, it's joint pos
            #print('joint_z is ', str(joint_z))
            reward = 1 - (joint_z)**2
            done = False
            return reward, done
        
        elif self.robot_name == 'pendulum':
            joint_theta = self.state_dict['joint_angles'][0]
            joint_velocity = self.state_dict['joint_velocities'][0]
            reward = angle_normalize(joint_theta)**2 + .1*joint_velocity**2 + .001*(self.last_action[0]**2)
            done = False
            return reward, done
        
        else:
            raise('Unknown robot name')
    
    def change_friction_coeffs(self, bodyUniqueId, physicsClientId, 
                               links_idx_list='all', 
                               lateral_coeff_list = 1., spinning_friction_list = 0.,
                               rolling_friction_list = 0.):
        
        import collections
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

    def reset(self):
        if self.init_state_id is None:
            p.resetSimulation(physicsClientId = self.physicsClientId)
            self.plane = p.loadURDF("plane.urdf", useFixedBase=True, physicsClientId = self.physicsClientId)
            self.robot = p.loadURDF(self.robot_urdf, basePosition = [0, 0, 1.], 
                                    baseOrientation = [0,0,1,1], useFixedBase=False,
                                    physicsClientId = self.physicsClientId)
            
            for bodyunqid in [self.plane, self.robot]:
                #break
                self.change_friction_coeffs(bodyUniqueId = bodyunqid, physicsClientId = self.physicsClientId, 
                                            links_idx_list='all', lateral_coeff_list = 1., spinning_friction_list = 1.,
                                            rolling_friction_list = 1.)
            
            self.get_robot_info() #Just to make sure that everything is updated
            
            #thickness = 0.04
            #slide_height = 5
            #create_side_sliders(slide_height = slide_height, slide_width = 0.06, wall_thickness = thickness, 
            #                    centerPosition = [-(1.27-thickness/2),0,slide_height/2+thickness], left_wall = True, 
            #                    physicsClientId = self.physicsClientId)
            #create_side_sliders(slide_height = slide_height, slide_width = 0.06, wall_thickness = thickness, 
            #                    centerPosition = [ (1.27-thickness/2),0,slide_height/2+thickness], right_wall = True,
            #                    physicsClientId = self.physicsClientId)
            self.init_state_id = p.saveState(physicsClientId = self.physicsClientId)
        else:
            p.restoreState(stateId = self.init_state_id, physicsClientId = self.physicsClientId)        
        
        
        self.viz_util = pyb_viz(physicsClientId = self.physicsClientId)
        deactivate_joint_motors(self.robot, joint_list = None, #[uu for uu in self.all_joints if not(uu in self.controllable_joints)], 
                                physicsClientId = self.physicsClientId, p=p)
        
        p.setRealTimeSimulation(0, physicsClientId = self.physicsClientId)
        p.setGravity(0, 0, -9.81, physicsClientId = self.physicsClientId)
        #p.setTimeStep(1.0 * self.time_step / self.t_res, physicsClientId = self.physicsClientId)
        
        
        if self.robot_name == 'monoped':
            for joint_id in self.controllable_joints:
                randstate = self.np_random.uniform(low=-0.5, high=0.5, size=(2, ))
                p.resetJointState(self.robot, joint_id, randstate[0], randstate[1]*0.0,
                                  physicsClientId = self.physicsClientId)
        elif self.robot_name == 'updown_box':
            for joint_id in self.controllable_joints:
                randpos = self.np_random.uniform(low=-.2, high=0.2) * 0.0
                randspeed = self.np_random.uniform(low=-1, high=+1) *0.0
                p.resetJointState(self.robot, joint_id, targetValue = randpos, 
                                  targetVelocity = randspeed, physicsClientId = self.physicsClientId)
        elif self.robot_name == 'pendulum':
            for joint_id in self.controllable_joints:
                randpos = self.np_random.uniform(low=-1, high=1) 
                randspeed = self.np_random.uniform(low=-0.5, high=+0.5)
                p.resetJointState(self.robot, joint_id, targetValue = randpos, 
                                  targetVelocity = randspeed, physicsClientId = self.physicsClientId)
        
        else:
            raise('Unknown robot name')
            
            
        self.state, self.state_dict, self.potential = self.get_state(robot = self.robot, physicsClientId = self.physicsClientId)
        
        self.configure_dispaly()

        return np.array(self.state)

    def render(self, mode='human', close=False):
        if self.gui:
            return 
        np_img_arr_h = self.im_renderer_h()
        np_img_arr_v = self.im_renderer_v()
        border = np.zeros((np_img_arr_h.shape[0], 10, 4))
        np_img_arr = np.concatenate([np_img_arr_v, border, np_img_arr_h], axis=1)
        self.vid_compiler.add_np_img(np_img_arr)
        return np_img_arr
    
    def compile_video(self, out_file='test.mp4', fps = 20):
        clip = self.vid_compiler(out_file=out_file, fps=fps)
        print('done video compiling!')
        return clip
        