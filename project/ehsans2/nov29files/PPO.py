from __future__ import unicode_literals
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data

from itertools import product
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.patches as patches
import types, io, os, time, math, inspect
from importlib import reload
import moviepy.editor as mpy
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import animation as animation_plt
from matplotlib import rc
from IPython import display
from IPython.display import HTML, clear_output
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

from utilities.render_utils import figure_compiler_video
from utilities.os_utils import mkdir_p
from utilities.general_utils import dict_append
from utilities.gym_utils import get_space_shape, get_angle
from gym_envs.legged import robotBulletEnv
import threading
from collections import OrderedDict 
import pandas as pd

import IPython.display
import PIL.Image
disp_np_im = lambda np_im: IPython.display.display(PIL.Image.fromarray(np_im.astype(np.uint8)))
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def mycumprodsum(my_delta, my_gamma):
    if torch.is_tensor(my_delta):
        my_delta = my_delta.to(dtype=torch.float)
        c = torch.arange(my_delta.numel()).to(my_delta)
        c = torch.pow(1./my_gamma, c)
        a = my_delta * c
        a = torch.cumsum(a, dim=0)
        return a / c
    else:
        my_delta = np.array(my_delta)
        c = np.arange(my_delta.size)
        c = np.power(1./my_gamma, c)
        a = my_delta * c
        a = np.cumsum(a)
        return a / c

def mycumprodsum_rev(my_delta, my_gamma): 
    if torch.is_tensor(my_delta):
        return mycumprodsum(my_delta.flip(0), my_gamma).flip(0)
    else:
        return mycumprodsum(my_delta[::-1], my_gamma)[::-1]
    
class tester_env_class():
    def __init__(self, Environment_Maker):
        test_env = Environment_Maker()

        test_env.exp_history=[]
        self.env = test_env
    
    def __call__(self, main_net, steps=200, reward_shaping = lambda x: x, render=False):
        
        test_env = self.env
        s = test_env.reset()
        test_env.last_render_time = 0
        test_env.sim_time = 0
        test_env.exp_history.append({'states':[], 'actions':[], 'rewards':[], 
                                     'next_states':[], 'std(a)':[], 'V(s)':[]})
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print('test device is: ', device)
        for t in range(steps):
            if render and (test_env.sim_time > (test_env.last_render_time + 0.05)):
                test_env.render()
                test_env.last_render_time = test_env.sim_time
                
            with torch.no_grad():
                s_tensor = torch.from_numpy(s).to(device=device, dtype=torch.float)
                #print('s_tensor device is ', s_tensor)
                s_tensor = s_tensor.unsqueeze(0)
                mean_act, logstd_act, value_state = main_net(s_tensor)
                mean_act.squeeze_(0), logstd_act.squeeze_(0), value_state.squeeze_(0)
                chosen_act = torch.randn_like(logstd_act) * torch.exp(logstd_act) + mean_act
            
            chosen_act_np = chosen_act.cpu().numpy()
            chosen_act_np = np.clip(chosen_act_np, -2, 2)
            s_prime, r, done, info = test_env.step(chosen_act_np)
            r = reward_shaping(r)
            done = False

            test_env.exp_history[-1]['states'].append(s)
            test_env.exp_history[-1]['actions'].append(chosen_act_np)
            test_env.exp_history[-1]['rewards'].append(r)
            test_env.exp_history[-1]['std(a)'].append(torch.exp(logstd_act).cpu().numpy())
            test_env.exp_history[-1]['V(s)'].append(value_state.cpu().numpy())
            test_env.exp_history[-1]['next_states'].append(s_prime)

            s = s_prime
            test_env.sim_time += test_env.time_step
            if done:
                break

class mean_std_val_net_v1(nn.Module):
    def __init__(self, state_space_shape, out_space_shape):
        super().__init__()
        self.state_space_shape = state_space_shape
        self.action_space_shape = out_space_shape
        inp_dim = np.prod(state_space_shape)
        out_dim = np.prod(out_space_shape)
        self.l1 = nn.Linear(inp_dim, 20)
        self.l1_relu = nn.LeakyReLU()
        self.l2 = nn.Linear(20, 100)
        self.l2_relu = nn.LeakyReLU()
        self.l4 = nn.Linear(100, 50)
        self.l4_relu = nn.LeakyReLU()
        self.l3 = nn.Linear(50, out_dim)

    def forward(self, input):
        input = input.reshape(-1, *self.state_space_shape)
        output = self.l1(input)
        output = self.l1_relu(output)
        output = self.l2(output)
        output = self.l2_relu(output)
        output = self.l4(output)
        output = self.l4_relu(output)
        output = self.l3(output)
        return output.reshape(-1, *self.action_space_shape)

class mean_std_val_net_v2(nn.Module):
    def __init__(self, state_space_shape, out_space_shape):
        super().__init__()
        self.state_space_shape = state_space_shape
        self.action_space_shape = out_space_shape
        inp_dim = np.prod(state_space_shape)
        out_dim = np.prod(out_space_shape)
        
        self.l1 = nn.Linear(inp_dim, 100)
        self.l1_relu = nn.ReLU()
        self.l2 = nn.Linear(100, out_dim)

    def forward(self, input):
        input = input.reshape(-1, *self.state_space_shape)
        output = self.l1(input)
        output = self.l1_relu(output)
        output = self.l2(output)
        return output.reshape(-1, *self.action_space_shape)

class mean_std_val_net_v3(nn.Module):
    def __init__(self, state_space_shape, out_space_shape, hidden_layers_units = [100], activation_fn = nn.ReLU):
        super().__init__()
        self.state_space_shape = state_space_shape
        self.out_space_shape = out_space_shape
        self.hidden_layers_units = hidden_layers_units
        inp_dim = np.prod(state_space_shape)
        out_dim = np.prod(out_space_shape)
        
        self.layers = []
        layer_idx = 0
        
        last_dim = inp_dim
        for _,units in enumerate(self.hidden_layers_units):
            lin_lay = nn.Linear(last_dim, units)
            act_lay = activation_fn()
            last_dim = units
            
            self.layers.append(lin_lay)
            self.layers.append(act_lay)
        
        lin_lay = nn.Linear(last_dim, out_dim)
        self.layers.append(lin_lay)
        self.seq_net = nn.Sequential(*self.layers)
        
        

    def forward(self, myinput):
        output = myinput.reshape(-1, *self.state_space_shape)
        output = self.seq_net(output)
        return output.reshape(-1, *self.out_space_shape)

def init_weights(m): 
    classname = m.__class__.__name__
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    #if classname.find('Linear') != -1:
    #    torch.nn.init.xavier_uniform_(m.weight)
    #    m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        #classname.find('Conv') != -1:
        #torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)


def run_ppo_experiment( N = 3, 
                        T = 32,
                        K = 10,
                        gamma = 0.9,
                        lamda = 1,
                        epsilon = 0.2,
                        num_loops = 10000,
                        batch_size = 32,
                        c_1 = 2,
                        c_2 = 0,
                        seperate_val_net = True,
                        reward_shaping = lambda r: (r+8)/8.,
                        action_sigma = None,
                        max_ep_len = 480,
                        learning_rate = 0.00001,
                        action_mean_transformation = lambda proposed_act_mean: 2 * tanh(proposed_act_mean),
                        action_std_transformation = lambda proposed_act_std: softplus(proposed_act_std),
                        neural_net_maker = lambda s_dim, out_dim: mean_std_val_net_v3(s_dim, out_dim, hidden_layers_units = [100]),
                        Environment_Maker = lambda : gym.make('Pendulum-v0'),
                        use_threads=False,
                        print_runningtime=False):

    try:
        mse_loss = torch.nn.MSELoss(reduction='elementwise_mean')
        softplus = nn.Softplus()
        tanh = nn.Tanh()
        #Creating the environments
        env_list = [Environment_Maker() for _ in range(N)]
        for i_thread in range(N):
            env = env_list[i_thread]
            env.last_state = env.reset()
            env.my_timer = 0
            env.exp_history = []
            env.need_reset = True
            env.corrupt_sim = False
            env.curr_episode = -1

        #Creating the network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        out_act_factor = 2 if action_sigma is None else 1
        if not seperate_val_net:
            main_nnet = neural_net_maker(state_space_shape = get_space_shape(env.observation_space),
                                            out_space_shape = [out_act_factor * np.prod(get_space_shape(env.action_space)) + 1]).to(device)
            #main_nnet = nn.DataParallel(main_nnet)
            main_nnet.apply(init_weights)
            main_optimizer = torch.optim.Adam(main_nnet.parameters(), lr=learning_rate)

            act_shape = [int(x) for _,x in enumerate(get_space_shape(env.action_space))]
            def main_net(x):
                net_out = main_nnet(x)
                act_means = net_out[...,0:np.prod(act_shape)].reshape(-1,*act_shape)
                act_means = action_mean_transformation(act_means)
                if action_sigma is None:
                    act_stds = net_out[...,np.prod(act_shape):(2*np.prod(act_shape))].reshape(-1,*act_shape)
                    act_stds = torch.log( action_std_transformation(act_stds) )
                else:
                    act_stds = torch.full_like(act_means, fill_value=float(np.log(action_sigma)), 
                                               dtype=torch.float, device=device, requires_grad=False)
                state_values = net_out[...,-1]
                return act_means, act_stds, state_values
        else:
            main_nnet = neural_net_maker(state_space_shape = get_space_shape(env.observation_space),
                                            out_space_shape = [out_act_factor * np.prod(get_space_shape(env.action_space))]).to(device)
            val_nnet = neural_net_maker(state_space_shape = get_space_shape(env.observation_space), out_space_shape = [1]).to(device)
            main_nnet.apply(init_weights)
            val_nnet.apply(init_weights)
            
            main_nnet = nn.DataParallel(main_nnet)
            val_nnet = nn.DataParallel(val_nnet)
            

            main_optimizer = torch.optim.Adam(list(main_nnet.parameters()) + list(val_nnet.parameters()), lr=learning_rate)

            act_shape = [int(x) for _,x in enumerate(get_space_shape(env.action_space))]

            my_softplus = nn.Softplus()
            def main_net(x):                
                net_out = main_nnet(x)
                val_out = val_nnet(x)
                act_means = net_out[...,0:np.prod(act_shape)].reshape(-1,*act_shape)
                act_means = action_mean_transformation(act_means)
                if action_sigma is None:
                    act_stds = net_out[...,np.prod(act_shape):(2*np.prod(act_shape))].reshape(-1,*act_shape)
                    act_stds = torch.log( action_std_transformation(act_stds) )
                else:
                    act_stds = torch.full_like(act_means, fill_value=float(np.log(action_sigma)), 
                                               dtype=torch.float, device=device, requires_grad=False)
                state_values = val_out[...,-1]
                return act_means, act_stds, state_values

        #Creating the test env
        test_env = tester_env_class(Environment_Maker = Environment_Maker)

        #Other stuff
        loss_hist = []
        other_hist = []
        episodic_hist = {}
        learning_vidmaker = figure_compiler_video()

        np.set_printoptions(precision=2, suppress=True)

        def run_step(kwargs):
            env = kwargs['env'] 
            s = kwargs['s']
            my_chosen_act = kwargs['my_chosen_act']
            my_mean_act = kwargs['my_mean_act']
            my_logstd_act = kwargs['my_logstd_act']
            my_value_state = kwargs['my_value_state']

            s_prime, r, done, info = env.step(my_chosen_act)
            r = reward_shaping(r)
            done = env.my_timer > (max_ep_len-2)

            sample_data = {'state': np.array(s, copy=True),
                           'next state': np.array(s_prime, copy=True),
                           'action': my_chosen_act,
                           'reward': r,
                           'done' : done,
                           'time' : env.my_timer,
                           'network action mean': my_mean_act,
                           'network action logstd': my_logstd_act,
                           'network state value': my_value_state}

            dict_append(env.exp_history[-1],sample_data)
            env.last_state = s_prime
            env.my_timer += 1

            if done:
                env.need_reset=True
            return

        #Training
        if print_runningtime:
            old_loop_time = time.time()
        try:
            for i_loop in range(num_loops):
                if print_runningtime:
                    new_loop_time = time.time()
                    if i_loop:
                        print('Full loop time: ', new_loop_time - old_loop_time)
                    old_loop_time = new_loop_time
                #Running the simulations using the old policy
                for env in env_list:
                    if env.need_reset:
                        #print('Just performed a reset')
                        env.last_state = env.reset()
                        env.my_timer = 0
                        env.curr_episode += 1
                        env.need_reset = False
                        env.corrupt_sim = False
                    env.t_start = env.my_timer
                    env.exp_history.append({})

                if print_runningtime:
                    step_time = 0
                    net_time = 0
                while True:
                    alive_env_idxs = [i for i,env in enumerate(env_list) if (env.my_timer < env.t_start + T) and not(env.need_reset)]
                    #print('len(alive_env_idxs) ', str(len(alive_env_idxs)), end=', ')
                    if not(len(alive_env_idxs)):
                        break

                    s_list = []
                    for i in alive_env_idxs:
                        s_list.append(env_list[i].last_state)
                    s_np = np.stack(s_list, axis=0)
                    if print_runningtime:
                        net_time -= time.time()

                    with torch.no_grad():
                        s_tensor = torch.from_numpy(s_np).to(device=device, dtype=torch.float)
                        mean_act, logstd_act, value_state = main_net(s_tensor)
                        chosen_act = torch.randn_like(logstd_act) * torch.exp(logstd_act) + mean_act
                    chosen_act_np = chosen_act.cpu().numpy()
                    mean_act_np, logstd_act_np, value_state_np = mean_act.cpu().numpy(), logstd_act.cpu().numpy(), value_state.cpu().numpy()
                    if print_runningtime:
                        net_time += time.time()
                        step_time -= time.time()
                    if use_threads:
                        th_list = []
                        for i_seq,i in enumerate(alive_env_idxs):
                            kwargs = dict(  env = env_list[i],
                                            s = s_np[i_seq],
                                            my_chosen_act = chosen_act_np[i_seq],
                                            my_mean_act = mean_act_np[i_seq],
                                            my_logstd_act = logstd_act_np[i_seq],
                                            my_value_state = value_state_np[i_seq])

                            th = threading.Thread(target=run_step, args=(kwargs,), daemon=True)
                            th_list.append(th)
                            th.start()

                        for th in th_list:
                            th.join()
                        for th in th_list:
                            del th

                    else:
                        for i_seq,i in enumerate(alive_env_idxs):
                            env = env_list[i]
                            s = s_np[i_seq]
                            my_chosen_act = chosen_act_np[i_seq]
                            my_mean_act = mean_act_np[i_seq]
                            my_logstd_act = logstd_act_np[i_seq]
                            my_value_state = value_state_np[i_seq]
                            s_prime, r, done, info = env.step(my_chosen_act)
                            r = reward_shaping(r)
                            done = env.my_timer > (max_ep_len-2)

                            sample_data = {'state': np.array(s, copy=True),
                                           'next state': np.array(s_prime, copy=True),
                                           'action': my_chosen_act,
                                           'reward': r,
                                           'done' : done,
                                           'time' : env.my_timer,
                                           'network action mean': my_mean_act,
                                           'network action logstd': my_logstd_act,
                                           'network state value': my_value_state}

                            if np.isnan(np.array(list(env.state) + [r])).any():
                                env.need_reset = True
                                env.corrupt_sim = True
                                print('Warning: Corrupted simulation happend...')
                                continue
                            env.last_state = s_prime
                            env.my_timer += 1

                            dict_append(env.exp_history[-1],sample_data)
                            if done:
                                env.need_reset=True
                                continue
                    if print_runningtime:
                        step_time += time.time()

                #time.sleep(2)
                #clear_output()
                if print_runningtime:
                    print('Net Time: ',net_time)
                    print('Step Time: ',step_time)

                data = []
                for i_thread in range(N):
                    env = env_list[i_thread]
                    if env.corrupt_sim:
                        continue
                    #Computing delta and advantage values
                    r_t = np.array(env.exp_history[-1]['reward'])
                    v_st = np.array(env.exp_history[-1]['network state value'])
                    v_st_plus_one = np.array(env.exp_history[-1]['network state value'][1:] + [env.exp_history[-1]['network state value'][-1]])

                    with torch.no_grad():
                        r_t_tensor = torch.from_numpy(r_t).to(device=device, dtype=torch.float)
                        v_st1_tensor = torch.from_numpy(v_st_plus_one).to(device=device, dtype=torch.float).reshape(-1)
                        v_st_tensor = torch.from_numpy(v_st).to(device=device, dtype=torch.float).reshape(-1)
                        delta = r_t_tensor +  gamma * v_st1_tensor - v_st_tensor
                        A_hat_t = mycumprodsum_rev(delta, lamda*gamma)

                        mu_old = torch.from_numpy(np.array(env.exp_history[-1]['network action mean'])).\
                                    to(device=device, dtype=torch.float)
                        logstd_old = torch.from_numpy(np.array(env.exp_history[-1]['network action logstd'])).\
                                        to(device=device, dtype=torch.float)
                        a_t = torch.from_numpy(np.array(env.exp_history[-1]['action'])).to(device=device, dtype=torch.float)
                        s_t = torch.from_numpy(np.array(env.exp_history[-1]['state'])).to(device=device, dtype=torch.float)
                        s_t1 = torch.from_numpy(np.array(env.exp_history[-1]['next state'])).to(device=device, dtype=torch.float)

                        log_pi_old =  (torch.pow(a_t - mu_old,2))/((-2) * torch.exp(2 * logstd_old)) - logstd_old 
                        log_pi_old = log_pi_old.sum(dim=1)  

                    data.append([A_hat_t, log_pi_old, a_t, s_t, s_t1, r_t_tensor, logstd_old])

                    for key, val in {'Reward': r_t.reshape(-1).tolist(),
                                     'Action std' : np.exp(logstd_old.cpu().numpy()).reshape(-1).tolist()}.items():
                        if not(key in episodic_hist):
                            episodic_hist[key] = {}
                        if not(env.curr_episode in episodic_hist[key]):
                            episodic_hist[key][env.curr_episode] = []
                        episodic_hist[key][env.curr_episode] += val

                if np.mean([int(env.corrupt_sim) for env in env_list])>0.9:
                    return dict(globals(), **locals())

                with torch.no_grad():
                    A_hat_t = torch.cat([d[0] for _,d in enumerate(data)], dim=0)
                    log_pi_old = torch.cat([d[1] for _,d in enumerate(data)], dim=0)
                    a_t = torch.cat([d[2] for _,d in enumerate(data)], dim=0)
                    s_t = torch.cat([d[3] for _,d in enumerate(data)], dim=0)
                    s_t1 = torch.cat([d[4] for _,d in enumerate(data)], dim=0)
                    r_t = torch.cat([d[5] for _,d in enumerate(data)], dim=0)
                    log_std_old_t = torch.cat([d[6] for _,d in enumerate(data)], dim=0)

                loss_hist.append({})
                other_hist.append({ **{'a-'+str(uu): (a_t[:,uu].cpu().numpy())  for uu in range(min(3, a_t.shape[1]))},
                                    **{'s-'+str(uu): (s_t[:,uu].cpu().numpy())  for uu in range(min(4, s_t.shape[1]))},
                                    **{'Action std': np.exp(log_std_old_t.cpu().numpy()), 'Reward': r_t.cpu().numpy()} 
                                  })

                for k in range(K):
                    ep_perm = torch.randperm(s_t.shape[0])

                    it_per_ep = np.ceil(s_t.shape[0]/batch_size)
                    for batch_iter in range(int(it_per_ep)):
                        main_nnet.zero_grad()
                        if seperate_val_net:
                            val_nnet.zero_grad()

                        with torch.no_grad():
                            curr_idx = ep_perm[batch_iter * batch_size: (batch_iter+1) * batch_size].to(device=device)
                            curr_s_t = torch.index_select(s_t , dim = 0 , index = curr_idx)
                            curr_s_t1 = torch.index_select(s_t1 , dim = 0 , index = curr_idx)
                            curr_a_t = torch.index_select(a_t , dim = 0 , index = curr_idx)
                            curr_r_t = torch.index_select(r_t , dim = 0 , index = curr_idx)
                            curr_logpi_old = torch.index_select(log_pi_old , dim = 0 , index = curr_idx)
                            curr_A_hat_t = torch.index_select(A_hat_t , dim = 0 , index = curr_idx)

                        curr_mu_new, curr_logstd_new, curr_v_st = main_net(curr_s_t)

                        log_pi_new = torch.pow(curr_a_t - curr_mu_new, 2) / (-2 * torch.exp(2 * curr_logstd_new)) - curr_logstd_new 
                        log_pi_new = log_pi_new.sum(dim=1)  

                        log_pi_diff = log_pi_new - curr_logpi_old
                        pi_ratio = torch.exp(log_pi_diff)

                        loss_clip_vec = torch.min(pi_ratio * curr_A_hat_t, 
                                                  torch.clamp(pi_ratio, min=1-epsilon, max=1+epsilon) * curr_A_hat_t)

                        loss_clip = loss_clip_vec.mean()

                        _, _, curr_v_st1 = main_net(curr_s_t1)
                        with torch.no_grad():
                            target_v = curr_v_st1 * gamma + curr_r_t
                        loss_VF = mse_loss(curr_v_st, target_v)

                        if c_2:
                            loss_Entropy = torch.sum(curr_logstd_new + torch.log(torch.tensor(2*np.pi)))
                            total_loss = c_1 * loss_VF - loss_clip - c_2 * loss_Entropy
                        else:
                            total_loss = c_1 * loss_VF - loss_clip

                        total_loss.backward()
                        main_optimizer.step()

                        dict_append(loss_hist[-1], {'Total loss': total_loss.detach().cpu().numpy(),
                                                    'Clip loss': loss_clip.detach().cpu().numpy(),
                                                    'Value loss' : loss_VF.detach().cpu().numpy()})            
                        if c_2:
                            dict_append(loss_hist[-1], {'Entropy loss' : loss_Entropy.detach().cpu().numpy()})

                if (i_loop)%100 ==0:
                    test_env(main_net, steps = max_ep_len,
                             reward_shaping = reward_shaping, render=False)
                    if i_loop:
                        clear_output(wait=True)
                    disp_np_im(test_env.env.render())
                    df_list = []
                    readytoprintdf = True
                    #learning_vidmaker.add_figure(test_env.env.figure)

                if (i_loop)%100 < 20:
                    np.set_printoptions(precision=4)
                    #print('Loop ' +str(i_loop) + '--> ' + str([key + ': ' + np.array2string(np.mean(val), precision=2) for key,val in loss_hist[-1].items()] + 
                    #                                         [key + ': ' + np.array2string(np.mean(val), precision=2) for key,val in other_hist[-1].items()]))
                    curr_row = {**{key:np.mean(val) for key,val in loss_hist[-1].items()}, **{key:np.mean(val) for key,val in other_hist[-1].items()}}
                    df_list.append(curr_row)
                elif readytoprintdf:
                    print(i_loop)
                    df = pd.DataFrame(df_list)
                    print(df)
                    readytoprintdf = False
                    
                    
                    

        except KeyboardInterrupt:
            print('Exiting Training, Hit interrupt one more time to exit')
        store_prefix = 'Experiments/OneLeggedFixedHopper'
        #Storing the results
        ctr=0
        while True:
            if not os.path.exists(store_prefix + '/' + str(ctr)):
                break
            ctr += 1


        store_folder = store_prefix + '/' + str(ctr)
        mkdir_p(store_folder)


        #Creating the loss figure
        fig = plt.figure(figsize=(20,10))
        ax = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        for my_label in loss_hist[0].keys():
            x = range(len(loss_hist))
            y = [np.mean(loss_hist[t][my_label]) for _,t in enumerate(x)]
            ax.plot(x, y, label=my_label)

        for my_label, my_val in episodic_hist.items():
            x = sorted(list(my_val.keys()))
            y = [np.mean(my_val[t]) for _,t in enumerate(x)]
            ax2.plot(x, y, label=my_label)

        ax.legend(), ax2.legend()
        ax.set_title('Mean Losses vs mini-episode')
        ax2.set_title('Statistics vs mini-episode')
        #show_inline_matplotlib_plots()
        fig.savefig(store_folder+'/LossPlot.png')
        clear_output()

        #Learning Samples Clip
        #learning_clip = learning_vidmaker(fps=1)
        #learning_clip.write_videofile(store_folder+'/LearningSamples.mp4')

        #The network
        store_material=main_nnet.state_dict()
        torch.save(store_material, store_folder+'/model.pth')

        #The animation
        test_env(main_net, steps = max_ep_len, reward_shaping = reward_shaping, render=True)
        gifclip = test_env.env.compile_video(out_file=store_folder+'/Trajectory.mp4', fps = 20)
        #%time test_env.env.ani.save(store_folder+'/Trajectory.gif', writer=animation_plt.PillowWriter(fps=10))
        #gifclip = mpy.VideoFileClip(store_folder+'/Trajectory.gif')
        #gifclip.write_videofile(store_folder+'/Trajectory.mp4')

        kwargs = {  'N' : N, 
                    'T' : T,
                    'K' : K,
                    'gamma' : gamma,
                    'lamda' : lamda,
                    'epsilon' : epsilon,
                    'Number of loops' : num_loops,
                    'Batch size' : batch_size,
                    'c_1' : c_1,
                    'c_2' : c_2,
                    'Seperate Value Network' : seperate_val_net,
                    'Reward Shaping' : reward_shaping,
                    'Action Sigma' : action_sigma,
                    'Max episode length' : max_ep_len,
                    'Learning rate' : learning_rate,
                    'Action mean transformation' : action_mean_transformation,
                    'Action std transformation' : action_std_transformation,
                    'Neural net maker' : neural_net_maker,
                    'Environment Maker' : Environment_Maker  }
        kwargs_str = {}
        for key, val in kwargs.items():
            if callable(val):
                kwargs_str[str(key)]=str(inspect.getsource(val))
            else:
                kwargs_str[str(key)]=str(val)

        for key, val in kwargs_str.items():
            print(key + ': ' + val, file=open(store_folder+'/Hyperparameters.txt', "w"))

        for key, val in kwargs_str.items():
            print(key + ': ' + val, file=open(store_prefix + '/ExperimentsIndex.txt', "a+"))
        print('----------------------', file=open(store_prefix + '/ExperimentsIndex.txt', "a+"))

        clear_output(wait=True)
        print('Resuls are stored in : ' + store_folder)
        return dict(globals(), **locals())
    except KeyboardInterrupt:
        return dict(globals(), **locals())