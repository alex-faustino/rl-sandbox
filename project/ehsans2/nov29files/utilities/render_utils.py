import pybullet as p
import numpy as np
import moviepy.editor as mpy
from moviepy.editor import VideoClip

class compiler_video():
    def __init__(self):
        self.imlist = []
    
    def add_figure(self, myfig):
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        buf.seek(0)
        im = np.array(Image.open(buf))
        buf.close()
        self.imlist.append(im)
    
    def add_np_img(self, np_img):
        self.imlist.append(np_img)
    
    def reset(self):
        for _ in range(len(self.imlist)):
            del(self.imlist[0])
        self.imlist=[]
    
    def __call__(self, out_file=None, fps=5): 
        clip = mpy.ImageSequenceClip(self.imlist, fps=fps)
        
        if not out_file is None:
            if out_file.endswith('gif'):
                clip.write_gif(out_file)
            else:
                clip.write_videofile(out_file)
        return clip
    

class image_renderer():
    def __init__(self, pixelWidth=640, pixelHeight=480, upAxisIndex = 2, physicsClientId = 0):
        self.physicsClientId = physicsClientId
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.aspect = pixelWidth / pixelHeight
        self.upAxisIndex = upAxisIndex
        
        self.camTargetPos = [0,0,0]
        self.camDistance = 4
        self.yaw = 0 #yaw angle in degrees left/right around up-axis
        self.pitch = -10 #pitch in degrees up/down
        self.roll = 0 #roll in degrees around forward vector
        
        self.set_view()
        self.set_projection()
    
    def set_view(self, camTargetPos = None, camDistance = None, yaw = None, pitch = None, roll = None):
        self.camTargetPos = camTargetPos if camTargetPos is not None else self.camTargetPos
        self.camDistance = camDistance if camDistance is not None else self.camDistance
        self.yaw = yaw if yaw is not None else self.yaw
        self.pitch = pitch if pitch is not None else self.pitch
        self.roll = roll if roll is not None else self.roll
        
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, self.yaw, 
                                                                     self.pitch, self.roll, self.upAxisIndex)
        
    def set_projection(self, fov=60, nearPlane=0.01, farPlane=100):
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov, self.aspect, nearPlane, farPlane)
    
    def __call__(self):
        img_arr = p.getCameraImage(self.pixelWidth, self.pixelHeight, self.viewMatrix, self.projectionMatrix, 
                                   shadow=1, lightDirection=[1,1,1],
                                   renderer = p.ER_BULLET_HARDWARE_OPENGL,
                                   physicsClientId = self.physicsClientId)
        w=img_arr[0] #width of the image, in pixels
        h=img_arr[1] #height of the image, in pixels
        rgb=img_arr[2] #color data RGB
        dep=img_arr[3] #depth data

        #reshape is needed
        np_img_arr = np.reshape(rgb, (h, w, 4))
        return np_img_arr
    
class figure_compiler_video():
    def __init__(self):
        self.imlist = []
        
    def add_figure(self, myfig):
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        buf.seek(0)
        im = np.array(Image.open(buf))
        buf.close()
        self.imlist.append(im)
    
    def __call__(self, out_size=5, fps=5, out_file=None,): 
        clip = mpy.ImageSequenceClip(self.imlist, fps=fps)
        
        if not out_file is None:
            if out_file.endswith('gif'):
                clip.write_gif(out_file)
            else:
                clip.write_videofile(out_file)
        return clip
    
    
def plt_render_general(self, exp_index=-1, act_per_frame = 1, create_mpy = False):
    assert False,'Not Implemeted Yet!'
    all_states = self.exp_history[exp_index]['states']
    all_actions = self.exp_history[exp_index]['actions']
    all_rewards = self.exp_history[exp_index]['rewards']
    other_keywords = [key for key in self.exp_history[exp_index].keys() if key not in ['states', 'actions', 'rewards', 'next_states']]

    if self.use_triangular_states:
        theta = (180./np.pi) * np.array(self.get_angle(cosine = [s[0] for _,s in enumerate(all_states)],
                                                       sine = [s[1] for _,s in enumerate(all_states)]))
        theta = (theta + 180) % 360
        omega = [s[2] for _,s in enumerate(all_states)]
    else:
        theta = [(180/np.pi) * s[0] for _,s in enumerate(all_states)]
        omega = [s[1] for _,s in enumerate(all_states)]
        
    torques = all_actions
    time_array = self.time_step * np.arange(len(theta))
    
    additional_rows = int(np.ceil(len(other_keywords) / 2.))
    plt_rows = (additional_rows + 4) if animation else (additional_rows + 2)
    if not(hasattr(self, 'plt_rows')):
        create_new_figure = True
    else:
        if plt_rows==self.plt_rows:
            create_new_figure = False
        else:
            create_new_figure = True                
    if not(hasattr(self,'figure')) or animation:
        create_new_figure = True

    self.plt_rows = plt_rows
    if create_new_figure:
        self.figure = plt.figure()

    self.figure.set_size_inches(plt_rows * 4, 2 * 4, forward=True)        

    if create_new_figure:
        self.act_ax = plt.subplot2grid((2, plt_rows), (0, 0))
        self.theta_ax = plt.subplot2grid((2, plt_rows), (0, 1))
        self.reward_ax = plt.subplot2grid((2, plt_rows), (1, 0))
        self.omega_ax = plt.subplot2grid((2, plt_rows), (1, 1))
        self.other_axes = [plt.subplot2grid((2, plt_rows), (int(uu % 2) , 2 + int(uu/2))) for uu,key in enumerate(other_keywords)]
    if animation:
        self.traj_ax = plt.subplot2grid((2, plt_rows), (0, additional_rows + 2), colspan=2, rowspan=2)

        self.traj_ax.set_xlim([-(self.l)*1.05, (self.l)*1.05])
        self.traj_ax.set_ylim([-(self.l)*1.05, (self.l)*1.05])

    plot_ax_list = [self.act_ax, self.theta_ax,
                    self.reward_ax, self.omega_ax] + self.other_axes
    plot_data_list = [torques, theta, all_rewards, omega] + [self.exp_history[exp_index][key] for _,key in enumerate(other_keywords)]
    plot_title_list = [r'$\tau$', r'$\theta^\circ$', 
                       r'$R$', r'$\omega$'] + other_keywords


    for i,curr_ax in enumerate(plot_ax_list):
        curr_data=plot_data_list[i]
        curr_ax.set_xlim([0,time_array[-1]])
        if i==1:
            curr_ax.set_ylim([0,360])
        else:
            if np.min(curr_data) == np.max(curr_data):
                curr_ax.set_ylim([np.min(curr_data) - 1, np.max(curr_data) + 1])
            else:
                curr_ax.set_ylim([np.min(curr_data)-0.05*np.abs(np.min(curr_data)),
                                  np.max(curr_data)+0.05*np.abs(np.max(curr_data))])
        curr_ax.set_title(plot_title_list[i], fontsize=16)

    if animation:
        self.acrobot, = self.traj_ax.plot([], [], 'o-', lw=2)
        traj_title = self.traj_ax.set_title('Trajectory', fontsize=16)

    if create_new_figure:
        self.plot_lines = []
        for i,curr_ax in enumerate(plot_ax_list):
            curr_line, = curr_ax.plot([], [], color='k')
            self.plot_lines.append(curr_line)

    def init():
        if animation:
            self.acrobot.set_data([], [])
            traj_title.set_text('Trajectory')
        for curr_line in self.plot_lines:
            curr_line.set_data([], [])
        return_list = [self.plot_lines[0], self.plot_lines[1], self.plot_lines[2], self.plot_lines[3]]
        if animation:
            return_list = [self.acrobot, traj_title] + return_list
        return return_list

    def animate(i):
        i = act_per_frame * i
        if animation:
            thisx = [0,  self.l * np.sin(theta[i]*np.pi/180)]
            thisy = [0, -self.l * np.cos(theta[i]*np.pi/180)]

            self.acrobot.set_data(thisx, thisy)
            traj_title.set_text('Trajectory (time= %.2f)'%(i*self.time_step))

        for j,curr_line in enumerate(self.plot_lines):
            curr_data = plot_data_list[j]
            curr_line.set_data(time_array[:i], curr_data[:i])

        return_list = [self.plot_lines[0], self.plot_lines[1], self.plot_lines[2], self.plot_lines[3]]
        if animation:
            return_list = [self.acrobot, traj_title] + return_list
        return return_list

    if animation:
        final_idx = int(len(theta)/act_per_frame)
        self.ani = animation_plt.FuncAnimation(self.figure, animate, np.arange(final_idx),
                                           interval=25, blit=True, init_func=init)
        if create_mpy:
            duration =  (final_idx * self.time_step)
            old_dpi = self.figure.get_dpi()
            self.figure.set_dpi(20)
            def make_frame(t):
                animate(int(t/self.time_step))
                return mplfig_to_npimage(self.figure)
            self.mpy_ani = VideoClip(make_frame, duration=duration)
            self.figure.set_dpi(old_dpi)

    else:
        animate(len(theta)-1)