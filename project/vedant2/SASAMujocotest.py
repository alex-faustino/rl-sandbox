#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer, load_model_from_path
import math
import os

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option gravity="0 0 0" integrator="RK4" timestep="1e-4" />
    <worldbody>
        <body name="satellite" pos="0 0 0.0">
            <joint axis="1 0 0" damping="0.0" name="BASE_DOF" pos="0 0 0" type="free"/>
            <geom mass="500.0" pos="0 0 0" rgba="0 0 1 1" size="2 2.3 3" type="box"/>
			<camera euler="0 0 0" fovy="60" name="rgb" pos="0 0 5.5"></camera>
		
        
            <body name="panelL0" pos="-5 0 0.0">
                <joint axis="1 0 0" stiffness="100" damping="0.0" name="panelL0_rotate_x" pos="1.5 0 0" type="hinge"/>
                <joint axis="0 1 0" stiffness="100" damping="0.0" name="panelL0_rotate_y" pos="1.5 0 0" type="hinge"/>
                <joint axis="0 0 1" stiffness="100" damping="0.0" name="panelL0_rotate_z" pos="1.5 0 0" type="hinge"/>
                <geom mass="20.0" rgba="0 1 0 1" size="2 0.15 3" type="box"/>
                
                <body name="panelL1" pos="-5 0 0.0">
                <joint axis="1 0 0" stiffness="100" damping="0.0" name="panelL1_rotate_x" pos="1.5 0 0" type="hinge"/>
                <joint axis="0 1 0" stiffness="100" damping="0.0" name="panelL1_rotate_y" pos="1.5 0 0" type="hinge"/>
                <joint axis="0 0 1" stiffness="100" damping="0.0" name="panelL1_rotate_z" pos="1.5 0 0" type="hinge"/>
                <geom mass="20.0" rgba="0 1 0 1" size="2 0.15 3" type="box"/>
            </body>
                
            </body>
            <body name="panelR0" pos="5 0 0.0">
                <joint axis="1 0 0" stiffness="100" damping="0.0" name="panelR0_rotate_x" pos="-1.5 0 0" type="hinge"/>
                <joint axis="0 1 0" stiffness="100" damping="0.0" name="panelR0_rotate_y" pos="-1.5 0 0" type="hinge"/>
                <joint axis="0 0 1" stiffness="100" damping="0.0" name="panelR0_rotate_z" pos="-1.5 0 0" type="hinge"/>
                <geom mass="20.0" rgba="1 0 0 1" size="2 0.15 3" type="box"/>
            
            <body name="panelR1" pos="5 0 0.0">
                <joint axis="1 0 0" stiffness="100" damping="0.0" name="panelR1_rotate_x" pos="-1.5 0 0" type="hinge"/>
                <joint axis="0 1 0" stiffness="100" damping="0.0" name="panelR1_rotate_y" pos="-1.5 0 0" type="hinge"/>
                <joint axis="0 0 1" stiffness="100" damping="0.0" name="panelR1_rotate_z" pos="-1.5 0 0" type="hinge"/>
                <geom mass="20.0" rgba="1 0 0 1" size="2 0.15 3" type="box"/>
            </body>
            
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor gear="0 0 0 1 0 0 " joint="BASE_DOF"/>
        <motor gear="0 0 0 0 1 0 " joint="BASE_DOF"/>
        <motor gear="0 0 0 0 0 1 " joint="BASE_DOF"/>
        
        <motor gear="10000 " joint="panelL0_rotate_x"/>
        <motor gear="10000 " joint="panelL0_rotate_y"/>
        <motor gear="10000 " joint="panelL0_rotate_z"/>
        
        <motor gear="10000 " joint="panelL1_rotate_x"/>
        <motor gear="10000 " joint="panelL1_rotate_y"/>
        <motor gear="10000 " joint="panelL1_rotate_z"/>
        
        <motor gear="10000 " joint="panelR0_rotate_x"/>
        <motor gear="10000 " joint="panelR0_rotate_y"/>
        <motor gear="10000 " joint="panelR0_rotate_z"/>
        
        <motor gear="10000 " joint="panelR1_rotate_x"/>
        <motor gear="10000 " joint="panelR1_rotate_y"/>
        <motor gear="10000 " joint="panelR1_rotate_z"/>
        
    </actuator>
</mujoco>
"""

#model = load_model_from_xml(MODEL_XML)
model = load_model_from_path(os.path.join(os.getcwd(),'SASA.xml'))
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
print(sim.model.body_inertia)
#while True:
while (True):
    
    sim.data.ctrl[0] = 0#math.cos(t / 10.) * 
    sim.data.ctrl[1] = 0
    sim.data.ctrl[2] = 0
    
    sim.data.ctrl[3] = math.cos(t / 10.) * 10 #math.cos(t / 10.) * 
    sim.data.ctrl[4] = math.cos(t / 10.) * 10
    sim.data.ctrl[5] = math.cos(t / 1.) * 10
    
    sim.data.ctrl[6] = 0#math.cos(t / 10.) * 
    sim.data.ctrl[7] = 0
    sim.data.ctrl[8] = 0
    
    sim.data.ctrl[9] = math.cos(t / 10.) * 10#math.cos(t / 10.) * 
    sim.data.ctrl[10] = math.cos(t / 10.) * 10
    sim.data.ctrl[11] = math.cos(t / 1.) * 10
    
    sim.data.ctrl[12] = 0#math.cos(t / 10.) * 
    sim.data.ctrl[13] = 0
    sim.data.ctrl[14] = 0
    '''
    dir(sim.model) : ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__pyx_vtable__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '_actuator_id2name', '_actuator_name2id', '_body_id2name', '_body_name2id', '_camera_id2name', '_camera_name2id', '_geom_id2name', '_geom_name2id', '_joint_id2name', '_joint_name2id', '_light_id2name', '_light_name2id', '_sensor_id2name', '_sensor_name2id', '_site_id2name', '_site_name2id', '_tendon_id2name', '_tendon_name2id', '_userdata_id2name', '_userdata_name2id', 'actuator_biasprm', 'actuator_biastype', 'actuator_cranklength', 'actuator_ctrllimited', 'actuator_ctrlrange', 'actuator_dynprm', 'actuator_dyntype', 'actuator_forcelimited', 'actuator_forcerange', 'actuator_gainprm', 'actuator_gaintype', 'actuator_gear', 'actuator_id2name', 'actuator_invweight0', 'actuator_length0', 'actuator_lengthrange', 'actuator_name2id', 'actuator_names', 'actuator_trnid', 'actuator_trntype', 'actuator_user', 'body_dofadr', 'body_dofnum', 'body_geomadr', 'body_geomnum', 'body_id2name', 'body_inertia', 'body_invweight0', 'body_ipos', 'body_iquat', 'body_jntadr', 'body_jntnum', 'body_mass', 'body_mocapid', 'body_name2id', 'body_names', 'body_parentid', 'body_pos', 'body_quat', 'body_rootid', 'body_subtreemass', 'body_user', 'body_weldid', 'cam_bodyid', 'cam_fovy', 'cam_ipd', 'cam_mat0', 'cam_mode', 'cam_pos', 'cam_pos0', 'cam_poscom0', 'cam_quat', 'cam_targetbodyid', 'cam_user', 'camera_id2name', 'camera_name2id', 'camera_names', 'dof_Madr', 'dof_armature', 'dof_bodyid', 'dof_damping', 'dof_frictionloss', 'dof_invweight0', 'dof_jntid', 'dof_parentid', 'dof_solimp', 'dof_solref', 'eq_active', 'eq_data', 'eq_obj1id', 'eq_obj2id', 'eq_solimp', 'eq_solref', 'eq_type', 'exclude_signature', 'geom_bodyid', 'geom_conaffinity', 'geom_condim', 'geom_contype', 'geom_dataid', 'geom_friction', 'geom_gap', 'geom_group', 'geom_id2name', 'geom_margin', 'geom_matid', 'geom_name2id', 'geom_names', 'geom_pos', 'geom_quat', 'geom_rbound', 'geom_rgba', 'geom_size', 'geom_solimp', 'geom_solmix', 'geom_solref', 'geom_type', 'geom_user', 'get_joint_qpos_addr', 'get_joint_qvel_addr', 'get_mjb', 'get_xml', 'hfield_adr', 'hfield_data', 'hfield_ncol', 'hfield_nrow', 'hfield_size', 'jnt_axis', 'jnt_bodyid', 'jnt_dofadr', 'jnt_limited', 'jnt_margin', 'jnt_pos', 'jnt_qposadr', 'jnt_range', 'jnt_solimp', 'jnt_solref', 'jnt_stiffness', 'jnt_type', 'jnt_user', 'joint_id2name', 'joint_name2id', 'joint_names', 'key_act', 'key_qpos', 'key_qvel', 'key_time', 'light_active', 'light_ambient', 'light_attenuation', 'light_bodyid', 'light_castshadow', 'light_cutoff', 'light_diffuse', 'light_dir', 'light_dir0', 'light_directional', 'light_exponent', 'light_id2name', 'light_mode', 'light_name2id', 'light_names', 'light_pos', 'light_pos0', 'light_poscom0', 'light_specular', 'light_targetbodyid', 'mat_emission', 'mat_reflectance', 'mat_rgba', 'mat_shininess', 'mat_specular', 'mat_texid', 'mat_texrepeat', 'mat_texuniform', 'mesh_face', 'mesh_faceadr', 'mesh_facenum', 'mesh_graph', 'mesh_graphadr', 'mesh_normal', 'mesh_vert', 'mesh_vertadr', 'mesh_vertnum', 'nM', 'na', 'name_actuatoradr', 'name_bodyadr', 'name_camadr', 'name_eqadr', 'name_geomadr', 'name_hfieldadr', 'name_jntadr', 'name_lightadr', 'name_matadr', 'name_meshadr', 'name_numericadr', 'name_sensoradr', 'name_siteadr', 'name_tendonadr', 'name_texadr', 'name_textadr', 'name_tupleadr', 'names', 'nbody', 'nbuffer', 'ncam', 'nconmax', 'nemax', 'neq', 'nexclude', 'ngeom', 'nhfield', 'nhfielddata', 'njmax', 'njnt', 'nkey', 'nlight', 'nmat', 'nmesh', 'nmeshface', 'nmeshgraph', 'nmeshvert', 'nmocap', 'nnames', 'nnumeric', 'nnumericdata', 'npair', 'nq', 'nsensor', 'nsensordata', 'nsite', 'nstack', 'ntendon', 'ntex', 'ntexdata', 'ntext', 'ntextdata', 'ntuple', 'ntupledata', 'nu', 'numeric_adr', 'numeric_data', 'numeric_size', 'nuser_actuator', 'nuser_body', 'nuser_cam', 'nuser_geom', 'nuser_jnt', 'nuser_sensor', 'nuser_site', 'nuser_tendon', 'nuserdata', 'nv', 'nwrap', 'opt', 'pair_dim', 'pair_friction', 'pair_gap', 'pair_geom1', 'pair_geom2', 'pair_margin', 'pair_signature', 'pair_solimp', 'pair_solref', 'qpos0', 'qpos_spring', 'sensor_adr', 'sensor_cutoff', 'sensor_datatype', 'sensor_dim', 'sensor_id2name', 'sensor_name2id', 'sensor_names', 'sensor_needstage', 'sensor_noise', 'sensor_objid', 'sensor_objtype', 'sensor_type', 'sensor_user', 'set_userdata_names', 'site_bodyid', 'site_group', 'site_id2name', 'site_matid', 'site_name2id', 'site_names', 'site_pos', 'site_quat', 'site_rgba', 'site_size', 'site_type', 'site_user', 'stat', 'tendon_adr', 'tendon_damping', 'tendon_frictionloss', 'tendon_id2name', 'tendon_invweight0', 'tendon_length0', 'tendon_lengthspring', 'tendon_limited', 'tendon_margin', 'tendon_matid', 'tendon_name2id', 'tendon_names', 'tendon_num', 'tendon_range', 'tendon_rgba', 'tendon_solimp_fri', 'tendon_solimp_lim', 'tendon_solref_fri', 'tendon_solref_lim', 'tendon_stiffness', 'tendon_user', 'tendon_width', 'tex_adr', 'tex_height', 'tex_rgb', 'tex_type', 'tex_width', 'text_adr', 'text_data', 'text_size', 'tuple_adr', 'tuple_objid', 'tuple_objprm', 'tuple_objtype', 'tuple_size', 'uintptr', 'userdata_id2name', 'userdata_name2id', 'userdata_names', 'vis', 'wrap_objid', 'wrap_prm', 'wrap_type']
    sim.model.body_mass
    sim.model.jnt_stiffness
    sim.model.jnt_stiffness
    print(sim.data.qpos)
    '''
    t += 1
    if(t==10000):
        sim.model.geom_size[0][0] = 5
        sim.model.geom_size[0][2] = 5
        print(sim.model.body_inertia)
        #sim.model.get_xml() = '<mujoco model="MuJoCo Model">\n    <compiler angle="radian" />\n    <option timestep="0.0001" gravity="0 0 0" integrator="RK4" />\n    <size njmax="500" nconmax="100" />\n    <worldbody>\n        <body name="satellite" pos="0 0 0">\n            <inertial pos="0 0 0" mass="500" diaginertia="2381.67 2166.67 1548.33" />\n            <joint name="BASE_DOF" type="free" />\n            <geom size="2 5 3" type="box" rgba="0 0 1 1" />\n            <camera name="rgb" pos="0 0 5.5" fovy="60" />\n            <body name="panelL0" pos="-5 0 0">\n                <inertial pos="0 0 0" mass="20" diaginertia="60.15 86.6667 26.8167" />\n                <joint name="panelL0_rotate_x" pos="1.5 0 0" axis="1 0 0" stiffness="100" />\n                <joint name="panelL0_rotate_y" pos="1.5 0 0" axis="0 1 0" stiffness="100" />\n                <joint name="panelL0_rotate_z" pos="1.5 0 0" axis="0 0 1" stiffness="100" />\n                <geom size="2 0.15 3" type="box" rgba="0 1 0 1" />\n                <body name="panelL1" pos="-5 0 0">\n                    <inertial pos="0 0 0" mass="20" diaginertia="60.15 86.6667 26.8167" />\n                    <joint name="panelL1_rotate_x" pos="1.5 0 0" axis="1 0 0" stiffness="100" />\n                    <joint name="panelL1_rotate_y" pos="1.5 0 0" axis="0 1 0" stiffness="100" />\n                    <joint name="panelL1_rotate_z" pos="1.5 0 0" axis="0 0 1" stiffness="100" />\n                    <geom size="2 0.15 3" type="box" rgba="0 1 0 1" />\n                </body>\n            </body>\n            <body name="panelR0" pos="5 0 0">\n                <inertial pos="0 0 0" mass="20" diaginertia="60.15 86.6667 26.8167" />\n                <joint name="panelR0_rotate_x" pos="-1.5 0 0" axis="1 0 0" stiffness="100" />\n                <joint name="panelR0_rotate_y" pos="-1.5 0 0" axis="0 1 0" stiffness="100" />\n                <joint name="panelR0_rotate_z" pos="-1.5 0 0" axis="0 0 1" stiffness="100" />\n                <geom size="2 0.15 3" type="box" rgba="1 0 0 1" />\n                <body name="panelR1" pos="5 0 0">\n                    <inertial pos="0 0 0" mass="20" diaginertia="60.15 86.6667 26.8167" />\n                    <joint name="panelR1_rotate_x" pos="-1.5 0 0" axis="1 0 0" stiffness="100" />\n                    <joint name="panelR1_rotate_y" pos="-1.5 0 0" axis="0 1 0" stiffness="100" />\n                    <joint name="panelR1_rotate_z" pos="-1.5 0 0" axis="0 0 1" stiffness="100" />\n                    <geom size="2 0.15 3" type="box" rgba="1 0 0 1" />\n                </body>\n            </body>\n        </body>\n    </worldbody>\n    <actuator>\n        <general joint="BASE_DOF" gear="0 0 0 1 0 0" />\n        <general joint="BASE_DOF" gear="0 0 0 0 1 0" />\n        <general joint="BASE_DOF" gear="0 0 0 0 0 1" />\n        <general joint="panelL0_rotate_x" gear="10000 0 0 0 0 0" />\n        <general joint="panelL0_rotate_y" gear="10000 0 0 0 0 0" />\n        <general joint="panelL0_rotate_z" gear="10000 0 0 0 0 0" />\n        <general joint="panelL1_rotate_x" gear="10000 0 0 0 0 0" />\n        <general joint="panelL1_rotate_y" gear="10000 0 0 0 0 0" />\n        <general joint="panelL1_rotate_z" gear="10000 0 0 0 0 0" />\n        <general joint="panelR0_rotate_x" gear="10000 0 0 0 0 0" />\n        <general joint="panelR0_rotate_y" gear="10000 0 0 0 0 0" />\n        <general joint="panelR0_rotate_z" gear="10000 0 0 0 0 0" />\n        <general joint="panelR1_rotate_x" gear="10000 0 0 0 0 0" />\n        <general joint="panelR1_rotate_y" gear="10000 0 0 0 0 0" />\n        <general joint="panelR1_rotate_z" gear="10000 0 0 0 0 0" />\n    </actuator>\n</mujoco>\n'
        
    sim.step(5000)
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break