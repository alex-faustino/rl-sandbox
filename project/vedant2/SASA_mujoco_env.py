# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:46:26 2018

@author: vedant2
"""
import mujoco_py
from mujoco_py import load_model_from_xml,load_model_from_path, MjSim, MjViewer
import math
import os
import numpy as np
import xml.etree.ElementTree


'''

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SASAEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'SASA.xml', 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent
'''

class SASAmujocoEnv:
    
    def __init__(self,Noise = None):

        #model = load_model_from_xml(MODEL_XML)
        model = load_model_from_path(os.path.join(os.getcwd(),'SASA.xml'))
        sim = MjSim(model)
        viewer = MjViewer(sim)
        t = 0
        dt = 10
        action_dim = njoints*2*3;
        observation_dim = 7;
        if (Noise != None):
            noise_dim = 6
        
    def step(self,a,render = False):
        
        action_noise = len(sim.data.ctrl)
        if (Noise != None):
            action_noise[0] = math.cos(t / 10.) * Dx#math.cos(t / 10.) * 
            action_noise[1] = math.cos(t / 10.) * Dy
            action_noise[2] = math.cos(t / 10.) * Dz
            
            action_noise[3] = math.cos(t / 10.) * Rx #math.cos(t / 10.) * 
            action_noise[4] = math.cos(t / 10.) * Ry
            action_noise[5] = math.cos(t / 1.) * Rz
    
        action_noise[self.noise_dim:self.noise_dim+len(a)] = a
        sim.data.ctrl = action_noise
        t += self.dt
        sim.step(dt)
        if(render):
            viewer.render()
        
    def reset(self,full_reset = False, sym_params):
        if full_reset:
            env_discp = xml.etree.ElementTree.parse('SASA.xml')
            root = env_discp.getroot()
            
            #main body
            a = root[2][0].attrib
            a['pos']
            
            #Panel_L0
            a = root[2][0][3].attrib
            a['pos']
            
            #Panel_L1
            a = root[2][0][3][4].attrib
            a['pos']
            
            #Panel_R0
            a = root[2][0][3].attrib
            a['pos']
            
            tree.write('SASA.xml')
            model = load_model_from_path(os.path.join(os.getcwd(),'SASA.xml'))
            sim = MjSim(model)
            viewer = MjViewer(sim)
        
        sim.model.bodymass = sym_params*sim.model.bodymass;
        sim.model.jnt_stiffness = sym_params*sim.model.jnt_stiffness;
        sim.model.geom_size[0][0] = 5
        sim.model.geom_size[0][2] = 5
        sim.data.qpos = sim.data.qpos
        sim.model.body_iquat = sim.model.body_iquat
        sim.model.body_ipos = sim.model.body_ipos
        sim.data.qvel = sim.data.qvel
        
        model = load_model_from_path(os.path.join(os.getcwd(),'SASA.xml'))
    '''
    dir(sim.model) : ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__pyx_vtable__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '_actuator_id2name', '_actuator_name2id', '_body_id2name', '_body_name2id', '_camera_id2name', '_camera_name2id', '_geom_id2name', '_geom_name2id', '_joint_id2name', '_joint_name2id', '_light_id2name', '_light_name2id', '_sensor_id2name', '_sensor_name2id', '_site_id2name', '_site_name2id', '_tendon_id2name', '_tendon_name2id', '_userdata_id2name', '_userdata_name2id', 'actuator_biasprm', 'actuator_biastype', 'actuator_cranklength', 'actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_dynprm', 'actuator_dyntype', 'actuator_forcelimited', 'actuator_forcerange', 'actuator_gainprm', 'actuator_gaintype', 'actuator_gear', 'actuator_id2name', 'actuator_invweight0', 'actuator_length0', 'actuator_lengthrange', 'actuator_name2id', 'actuator_names', 'actuator_trnid', 'actuator_trntype', 'actuator_user', 'body_dofadr', 'body_dofnum', 'body_geomadr', 'body_geomnum', 'body_id2name', 'body_inertia', 'body_invweight0', 'body_ipos', 'body_iquat', 'body_jntadr', 'body_jntnum', 'body_mass', 'body_mocapid', 'body_name2id', 'body_names', 'body_parentid', 'body_pos', 'body_quat', 'body_rootid', 'body_subtreemass', 'body_user', 'body_weldid', 'cam_bodyid', 'cam_fovy', 'cam_ipd', 'cam_mat0', 'cam_mode', 'cam_pos', 'cam_pos0', 'cam_poscom0', 'cam_quat', 'cam_targetbodyid', 'cam_user', 'camera_id2name', 'camera_name2id', 'camera_names', 'dof_Madr', 'dof_armature', 'dof_bodyid', 'dof_damping', 'dof_frictionloss', 'dof_invweight0', 'dof_jntid', 'dof_parentid', 'dof_solimp', 'dof_solref', 'eq_active', 'eq_data', 'eq_obj1id', 'eq_obj2id', 'eq_solimp', 'eq_solref', 'eq_type', 'exclude_signature', 'geom_bodyid', 'geom_conaffinity', 'geom_condim', 'geom_contype', 'geom_dataid', 'geom_friction', 'geom_gap', 'geom_group', 'geom_id2name', 'geom_margin', 'geom_matid', 'geom_name2id', 'geom_names', 'geom_pos', 'geom_quat', 'geom_rbound', 'geom_rgba', 'geom_size', 'geom_solimp', 'geom_solmix', 'geom_solref', 'geom_type', 'geom_user', 'get_joint_qpos_addr', 'get_joint_qvel_addr', 'get_mjb', 'get_xml', 'hfield_adr', 'hfield_data', 'hfield_ncol', 'hfield_nrow', 'hfield_size', 'jnt_axis', 'jnt_bodyid', 'jnt_dofadr', 'jnt_limited', 'jnt_margin', 'jnt_pos', 'jnt_qposadr', 'jnt_range', 'jnt_solimp', 'jnt_solref', 'jnt_stiffness', 'jnt_type', 'jnt_user', 'joint_id2name', 'joint_name2id', 'joint_names', 'key_act', 'key_qpos', 'key_qvel', 'key_time', 'light_active', 'light_ambient', 'light_attenuation', 'light_bodyid', 'light_castshadow', 'light_cutoff', 'light_diffuse', 'light_dir', 'light_dir0', 'light_directional', 'light_exponent', 'light_id2name', 'light_mode', 'light_name2id', 'light_names', 'light_pos', 'light_pos0', 'light_poscom0', 'light_specular', 'light_targetbodyid', 'mat_emission', 'mat_reflectance', 'mat_rgba', 'mat_shininess', 'mat_specular', 'mat_texid', 'mat_texrepeat', 'mat_texuniform', 'mesh_face', 'mesh_faceadr', 'mesh_facenum', 'mesh_graph', 'mesh_graphadr', 'mesh_normal', 'mesh_vert', 'mesh_vertadr', 'mesh_vertnum', 'nM', 'na', 'name_actuatoradr', 'name_bodyadr', 'name_camadr', 'name_eqadr', 'name_geomadr', 'name_hfieldadr', 'name_jntadr', 'name_lightadr', 'name_matadr', 'name_meshadr', 'name_numericadr', 'name_sensoradr', 'name_siteadr', 'name_tendonadr', 'name_texadr', 'name_textadr', 'name_tupleadr', 'names', 'nbody', 'nbuffer', 'ncam', 'nconmax', 'nemax', 'neq', 'nexclude', 'ngeom', 'nhfield', 'nhfielddata', 'njmax', 'njnt', 'nkey', 'nlight', 'nmat', 'nmesh', 'nmeshface', 'nmeshgraph', 'nmeshvert', 'nmocap', 'nnames', 'nnumeric', 'nnumericdata', 'npair', 'nq', 'nsensor', 'nsensordata', 'nsite', 'nstack', 'ntendon', 'ntex', 'ntexdata', 'ntext', 'ntextdata', 'ntuple', 'ntupledata', 'nu', 'numeric_adr', 'numeric_data', 'numeric_size', 'nuser_actuator', 'nuser_body', 'nuser_cam', 'nuser_geom', 'nuser_jnt', 'nuser_sensor', 'nuser_site', 'nuser_tendon', 'nuserdata', 'nv', 'nwrap', 'opt', 'pair_dim', 'pair_friction', 'pair_gap', 'pair_geom1', 'pair_geom2', 'pair_margin', 'pair_signature', 'pair_solimp', 'pair_solref', 'qpos0', 'qpos_spring', 'sensor_adr', 'sensor_cutoff', 'sensor_datatype', 'sensor_dim', 'sensor_id2name', 'sensor_name2id', 'sensor_names', 'sensor_needstage', 'sensor_noise', 'sensor_objid', 'sensor_objtype', 'sensor_type', 'sensor_user', 'set_userdata_names', 'site_bodyid', 'site_group', 'site_id2name', 'site_matid', 'site_name2id', 'site_names', 'site_pos', 'site_quat', 'site_rgba', 'site_size', 'site_type', 'site_user', 'stat', 'tendon_adr', 'tendon_damping', 'tendon_frictionloss', 'tendon_id2name', 'tendon_invweight0', 'tendon_length0', 'tendon_lengthspring', 'tendon_limited', 'tendon_margin', 'tendon_matid', 'tendon_name2id', 'tendon_names', 'tendon_num', 'tendon_range', 'tendon_rgba', 'tendon_solimp_fri', 'tendon_solimp_lim', 'tendon_solref_fri', 'tendon_solref_lim', 'tendon_stiffness', 'tendon_user', 'tendon_width', 'tex_adr', 'tex_height', 'tex_rgb', 'tex_type', 'tex_width', 'text_adr', 'text_data', 'text_size', 'tuple_adr', 'tuple_objid', 'tuple_objprm', 'tuple_objtype', 'tuple_size', 'uintptr', 'userdata_id2name', 'userdata_name2id', 'userdata_names', 'vis', 'wrap_objid', 'wrap_prm', 'wrap_type']
    sim.model.body_mass
    sim.model.jnt_stiffness
    sim.model.jnt_stiffness
    print(sim.data.qpos)
    '''    
        
        
    