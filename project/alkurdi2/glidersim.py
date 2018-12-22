from gym import core, spaces
from gym.utils import seeding
import numpy as np
from collections import namedtuple
import scipy as sp
__copyright__ = "Copyright 2018, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Abdul Alkurdi"]
__author__ = "Abdul Alkurdi alkurdi2@illinois.edu>"

# SOURCE:
# https://github.com/AbdulAlkurdi/AE598RL

class GliderSim(core.Env):
    
    """
    GliderSim is a aircraft flight simulator. Aircraft chosen is Blixer 2 with flight data obtained from report published by Team Pegasus from MIT: http://stanford.edu/~ichter/aa241x/ProblemSet1.pdf
    
    **STATE:**
    [x, y, z, u, v, w, (acceleration_w)].
    
    **ACTIONS:**
    [+15 0 -15] adding degrees to bank angle
    **REFERENCE:**
    The simulator is inspired by the desire to replicate following work:
    Reddy, G., Wong-Ng, J., Celani, A., Sejnowski, T. J., & Vergassola, M. (2018). Glider soaring via reinforcement learning in the field. Nature, 562(7726), 236.
    Reddy, G., Celani, A., Sejnowski, T. J., & Vergassola, M. (2016). Learning to soar in turbulent environments. Proceedings of the National Academy of Sciences, 113(33), E4877-E4884.    
    .. seealso::
    This book was used to aid with the modeling segment.
    Hull, D. G. (2007). Fundamentals of airplane flight mechanics (p. 15). Berlin: Springer.
    .. seealso::
    *version*
    This version runs well without wind yet. 
    """
   
    dt = 1
    
    state_class=namedtuple("state",["pos","vel","acc"])
    r_class=namedtuple("pos",['x','y','z','mu','gamma','phi']) # mu : bank, gamma : pitch, phi : yaw
    rd_class=namedtuple("vel",['x','y','z','mu','gamma','phi'])
    rdd_class=namedtuple("acc",['x','y','z','mu','gamma','phi'])
    #%% Sim Properties
    
    #Field properties
    max_x = 1000 #meters
    max_y = 1000
    max_z = 2000
    
    rho = 1.2 #  kg/m^3 fixed density
    g = 9.81
    
    ## Aircraft properties
    glide_angle = 5.193/180*np.pi # 5.2/180*np.pi #degrees
    v0 = 6.28      #m/s (~26ft/s) v stall around 25ft/s. velocity should be around 8 but reduced to make env smaller
    ld_ratio = 11.09 #lift to drag
    clmax = 1.18 #taking cl as clmax through flight because we're assuming a fixed angle of attack at all flight conditions after trimming
    mass = 0.57986 #kg bixler with smaller battery
    S = 0.2 # Wing Area (both wings)
    wing_loading = 21.8 #wing loading unit N/m2
    wing_span = 1.5 #1500mm
    moment_arm = (1.5/2)/2 # d between cg of wing and cg of aircraft
    Ixx, Iyy, Izz = 0.042, 0.029, 0.066 ## Ixx kg-m^2
    aircraft_metadata = namedtuple("aircraft_data",
                             ["S","wing_loading","moment_arm",
                              "mass","clmax","ld_ratio",
                              "v0","glide_angle", 'Ixx', 'Iyy', 'Izz'])
    bixler_meta = aircraft_metadata(S, wing_loading, moment_arm,
                              mass, clmax, ld_ratio,
                              v0, glide_angle, Ixx, Iyy, Izz)
    
    # Wind
    wind_model = namedtuple("Gust",["pmean", "pmu", "umean", "umu", "period"])
    wind_rolls = 1 #1:true, this dictates weather controller corrects for gust rolling
    num_of_thermals = 4 #specified number of "eyes" for the gust. where each it's on system.
    # Actions
    actions_num = 3 # +15, 0 , -15 dmu
    avail_bank = [-15/180*np.pi , 0 , 15/180*np.pi]
    
    # States & Observations
    roll_rate_bins = 3
    bank_angle_bins = 5 
    max_bank = [-30/180*np.pi, 30/180*np.pi]
    a_lim = [-5*g, 5*g]
    v_lim = [[-45, 45],[-20, 20]] # vlimx, vlimz 
    z_lim = [0, 6000] # limit of altitude. 
    #%% class(Sim) definitions
    
    def __init__(self):
        self.viewer = None
        
        self.observation_space = spaces.MultiDiscrete([self.bank_angle_bins, self.roll_rate_bins])
        
        self.action_space = spaces.Discrete(self.actions_num)
        self.state = None
        self.Gust = [0] * self.num_of_thermals
        self.seed()
        self.Time = 0
        self.zvel = [0,0] # zvel is the vertical velocity generated from the gust. 
        self.zvel_left , self.zvel_right = self.zvel    
        self.moment = 0 
        self.tester = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
     #@property
    def reset(self):
        self.moment = 0 
        self.Time = 0

        
        v0 = self.v0
        x0, y0, z0 = 0, self.max_y/2, self.max_z #always starts in the middle since gust is random
        mu, gamma, phi = 0., -self.glide_angle, 0
        xd0, yd0, zd0, mud0, gammad0, phid0 = v0*np.cos(gamma), 0., v0*np.sin(gamma), 0., 0., 0.
        xdd0, ydd0, zdd0, mudd0, gammadd0, phidd0 = 0., 0., 0., 0., 0., 0.
        
        r = self.r_class(x0,y0,z0, mu, gamma, phi)
        rd = self.rd_class(xd0, yd0, zd0, mud0, gammad0, phid0)
        rdd = self.rdd_class(xdd0, ydd0, zdd0, mudd0, gammadd0, phidd0)
        
        self.state = self.state_class(r, rd, rdd)
        
        pmean = np.zeros( [self.num_of_thermals, 2] )
        pmu = np.zeros(self.num_of_thermals)
        umean = np.zeros_like(pmu)
        umu = np.zeros_like(pmu)
        period = np.zeros_like(pmu)
        
        for i in range(self.num_of_thermals):
            
            pmean[i] = (np.random.rand()*self.max_x, np.random.rand()*self.max_y)
            pmu[i] = (np.random.rand()*0.15*self.max_x + 0.15 * self.max_x)
            umean[i] = (np.random.rand()*1 + 1)
            umu[i] = (np.random.rand()*2)
            period [i] = np.random.rand()*20 + 30 
            self.Gust[i] = self.wind_model(pmean[i], pmu[i], umean[i], umu[i], period[i])
            
        return self._get_ob() 
    def get_wind(self, pos, heading):
        t = self.Time
        zvel_left , zvel_right = 0, 0
        test=[]
        for i in range(self.num_of_thermals):
            gust = self.Gust[i]
            
            lwp = (pos.x + self.moment_arm * np.cos(heading - np.pi), pos.y + self.moment_arm * np.sin(heading - np.pi)) # wing_right_pos
            rwp = (pos.x + self.moment_arm * np.cos(heading + np.pi), pos.y + self.moment_arm * np.sign(heading - np.pi)) # wing_left_pos

            l_d = np.linalg.norm([lwp[0] - gust.pmean[0], lwp[1] - gust.pmean[1]]) # distance from wing to eye of gust
            r_d = np.linalg.norm([rwp[0] - gust.pmean[0], rwp[1] - gust.pmean[1]])  # distance from wing to eye of gust
            
            zvel_left  += min(sp.stats.norm(scale = gust.pmu).cdf(l_d), 1-sp.stats.norm(scale = gust.pmu).cdf(l_d)) * 2 * (gust.umean + np.sin( t / gust.period) * gust.umu)
            zvel_right += min(sp.stats.norm(scale = gust.pmu).cdf(r_d), 1-sp.stats.norm(scale = gust.pmu).cdf(r_d)) * 2 * (gust.umean + np.sin( t / gust.period) * gust.umu)
            
            #test.append( 1-sp.stats.norm(scale = gust.pmu).cdf(l_d))
            #gaussian model
        self.tester2 = test

        # intensity = min(sp.stats.norm(1,2).cdf(5),1 - sp.stats.norm(1, 2).cdf(5))
        self.zvel = zvel_left, zvel_right 
        
        
        #self.zvel = 0.1*np.random.randint(2), 0.15 *np.random.randint(2)
        return (self.zvel)
    
    def step(self, action): # action 0:15deg bank left = delta mu
        
        g=self.g
        rho=self.rho
        
        aircraft = self.bixler_meta
        Ixx, Iyy, Izz = aircraft.Ixx, aircraft.Iyy, aircraft.Izz
        S, cl , M= aircraft.S, aircraft.clmax, aircraft.mass
        a_lim, v_lim = self.a_lim, self.v_lim 
        d_bank = self.avail_bank[action]
        moment_arm = self.moment_arm
        
        s = self.state
        dt = self.dt
                
        saccx, saccy, saccz, saccgamma, saccmu, saccphi = 0, 0, 0, 0, 0, 0
        svelx, svely, svelz, svelgamma, svelmu, svelphi = 0, 0 ,0, 0, 0, 0
        sposx, sposy, sposz, sposgamma, sposmu, sposphi = 0, 0 ,0, 0, 0, 0

        w_left, w_right = self.get_wind(s.pos,s.pos.phi) #heading should be the angle in the xy plane
        
        vel_mag_l = np.sqrt( w_left**2 + s.vel.x**2 + s.vel.y**2  )
        vel_mag_r = np.sqrt( w_right**2 + s.vel.x**2 + s.vel.y**2)
        
        self.velmag = np.sqrt(s.vel.x**2 + s.vel.y**2)
        
        L_l = 1/2 * rho * S/2 * vel_mag_l**2 * cl 
        L_r = 1/2 * rho * S/2 * vel_mag_r**2 * cl  
        L = L_r + L_l
        L = bound(L, 10)
        D = L / 11
        self.Lift = L
        self.Drag = D
        ds = np.sign(s.vel.x) # insures drag always opposite of velocity direction. 
        
        moment = ( L_l - L_r ) * moment_arm 
        
        self.tester = [w_left, w_right, moment, self.Lift/(self.mass*self.g)]
        
        mu_aug = bound(s.pos.mu + d_bank, self.max_bank) # this is input mu.         
        
        getacc = {-30/180*np.pi : 5.715 * np.sin(-30/180*np.pi),
                 +30/180*np.pi : 5.715 * np.sin(+30/180*np.pi),
                 -15/180*np.pi : 5.715 * np.sin(+-15/180*np.pi),
                 +15/180*np.pi : 5.715 * np.sin(+15/180*np.pi),
                 +00/180*np.pi : 5.715 * np.sin(+00/180*np.pi)}
        saccy = getacc[mu_aug]
        saccz = (+np.cos(mu_aug) * (L * np.cos(s.pos.gamma) - D * np.sin(s.pos.gamma)*ds)) / M - g
        saccgamma = 0
        saccmu = moment / Ixx   
        saccphi = 0
        
        acc = self.rdd_class(bound(saccx, a_lim), bound(saccy, a_lim),
                             bound(saccz, a_lim), saccmu, saccgamma, saccphi)
        
        #------------------------------------------
        
        
        v = self.v0
        
        dphi = np.arctan2(saccy*dt,v)
        
        sposphi = s.pos.phi + dphi
        velx = v * np.cos(sposphi) 
        
        vely = v * np.sin(sposphi) 
        
        svelmu = s.vel.mu + acc.mu * dt 
        svelz  = s.vel.z + acc.z * dt
        svelphi = dphi
        svelgamma = s.vel.gamma + acc.gamma * dt
        
        
        vel = self.rd_class(bound(velx ,v_lim[0]), bound(vely, v_lim[0]),
                                  bound(svelz,v_lim[1]), svelmu, svelgamma, svelphi)
        

        
        sposmu = mu_aug 
        mu_aug = bound(sposmu, self.max_bank) #updating mu with mu_final
        
        sposx = s.pos.x + velx * dt
        sposy = s.pos.y + vely * dt
        
        sposz = s.pos.z + vel.z * dt + 1/2 * dt**2 * acc.z
        sposphi = (s.pos.phi + dphi) 
        sposgamma = s.pos.gamma + vel.gamma*dt + acc.gamma/2*dt**2
        
        
        
        z_aug = bound(sposz, self.z_lim)
        pos = self.r_class(sposx, sposy, z_aug, mu_aug, sposgamma, sposphi)
        new_state = self.state_class(pos, vel, acc)
        
        self.state = new_state
        reward = self._getr()
        self.Time += dt 
        self.moment = moment
        return self._get_ob(), reward
        
    def _get_ob(self):
        
        bins = np.array([-90, -22.5,  -7.5,   7.5,  22.5,  90])
        digit_state1 = np.digitize(self.state.pos.mu, bins).item()
        
        bins2 = np.array([-9999, -0.0009, 0.0009, 9999])
        digit_state2 = np.digitize(self.moment, bins2).item()
        
        digit_state = [digit_state1, digit_state2]
        
        return digit_state 
    def _getr(self):
        az = self.state.acc.z
        
        if az > 0.005:
            reward = 1
        elif az < - 0.05 :
            reward = -1
        else:
            reward = 0
        return reward
#%%
def wrap(angle):
    while 2* np.pi < angle: # wrap angle between -180 and 180
        angle = angle - 2*np.pi
    while angle < 0: 
        angle = angle + 2*np.pi
        
    return angle
#%%
def bound(x,x_limit):
    if type(x_limit) == int:
        x_limit = 0, x_limit
    return min(max(x, x_limit[0]), x_limit[1])
  
def getplots():
    x=[]
    y=[]
    z=[]
    
    for v in big_state:
        x.append(v.pos.x)
        y.append(v.pos.y)
        z.append(v.pos.z)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(x, y, z, label='random policy')
    ax.legend()
    return fig  