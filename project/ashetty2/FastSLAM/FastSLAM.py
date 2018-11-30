from IPython import display
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import matplotlib.pyplot as plt

# Fast SLAM covariance
Q = np.diag([0.1, np.deg2rad(1.0)])**2
R = np.diag([1.0, np.deg2rad(20.0)])**2

#  Simulation parameter
Qsim = 0*np.diag([0.3, np.deg2rad(2.0)])**2
Rsim = np.diag([0.5, np.deg2rad(10.0)])**2
#OFFSET_YAWRATE_NOISE = 0.05*np.random.uniform(-1,1,1)[0]
#OFFSET_YAWRATE_NOISE = 0.05*(1 if np.random.random() < 0.5 else -1)
OFFSET_YAWRATE_NOISE = 0.001*0

BOX_HALF_SIZE = 20.0
TRAJ_AMP = 0.5*0
TRAJ_FREQ = 2.0
INIT_YAW = TRAJ_AMP*TRAJ_FREQ
TRAJ_XLIM = [-1.0, 1.0]
TRAJ_YLIM = [-TRAJ_AMP-1.0, TRAJ_AMP+1.0]
PLT_XLIM = [TRAJ_XLIM[0]-BOX_HALF_SIZE, TRAJ_XLIM[1]+BOX_HALF_SIZE]
PLT_YLIM = [TRAJ_YLIM[0]-BOX_HALF_SIZE, TRAJ_YLIM[1]+BOX_HALF_SIZE]
TIME_OFFSET = 50.0

LM_BOX_SIZE = 8.0
N_LM = 5

MAX_ACTION_DEG = 5.0  #maximum possible action per step
MAX_RANGE = 40.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 1  # number of particle
NTH = N_PARTICLE / 1.0  # Number of particle for re-sampling

BIN_SIZE = 15.0  #degrees
FOV = 30.0  # field of view of range sensor [deg]

ALPHA = 0.1

class Particle:

	def __init__(self, N_LM):
		self.w = 1.0 / N_PARTICLE
		self.x = 0.0
		self.y = 0.0
		self.yaw = INIT_YAW
		self.P = np.eye(3)
		# landmark x-y positions
		self.lm = np.matrix(np.zeros((N_LM, LM_SIZE)))
		# landmark position covariance
		self.lmP = np.matrix(np.zeros((N_LM * LM_SIZE, LM_SIZE)))
		self.lmSeen = []

class FastSLAM(gym.Env):


	DT = 0.1  # time tick [s]

	def __init__(self):
		self.state = None
		self.reward_type = 'Error'

	def step(self, action):

		action = np.clip(action, -np.deg2rad(MAX_ACTION_DEG), np.deg2rad(MAX_ACTION_DEG))

		self.theta += action

		self.time += self.DT
		self.u = self.calc_input(self.time)

		#main particle filter loop
		self.xTrue, self.z, self.xDR, self.ud = self.observation(self.xTrue, self.xDR, self.u, self.RFID)
		self.particles = self.fast_slam2(self.particles, self.ud, self.z)
		self.xEst = self.calc_final_state(self.particles)
		self.xEst = self.xTrue
		self.x_state = self.xEst[0: STATE_SIZE]
		#add state varaibles to info
		info = {}
		info["estimated"] = self.x_state
		info["dead_reckoning"] = self.xDR
		info["true"] = self.xTrue

		#store history data
		self.hxEst = np.hstack((self.hxEst, self.x_state))
		self.hxDR = np.hstack((self.hxDR, self.xDR))
		self.hxTrue = np.hstack((self.hxTrue, self.xTrue))

		#calculate reward based on pose error
		reward = self.get_reward_LM()		

		#get state vector
		self.state = self.get_obs_fixedLM()
		#self.state = self.get_obs()

		return self.state, reward, False, info
		#return info


	def reset(self):

		self.time = 0.0

		self.theta = 0.0

		self.xEst = np.matrix(np.zeros((STATE_SIZE, 1)))		
		self.xTrue = np.matrix(np.zeros((STATE_SIZE, 1)))
		self.xDR = np.matrix(np.zeros((STATE_SIZE, 1)))  # Dead reckoning
		self.xEst[2], self.xTrue[2], self.xDR[2] = INIT_YAW, INIT_YAW, INIT_YAW
		self.x_state = self.xEst[0: STATE_SIZE]

		self.hxEst = self.xEst
		self.hxTrue = self.xTrue
		self.hxDR = self.xTrue

		count_vis = 0
		while count_vis==0:
			#arrange RFID
			find_lmc = True
			while find_lmc:
				lmcx = np.random.uniform(PLT_XLIM[0]+LM_BOX_SIZE/2.0, PLT_XLIM[1]-LM_BOX_SIZE/2.0,1)
				lmcy = np.random.uniform(PLT_YLIM[0]+LM_BOX_SIZE/2.0, PLT_YLIM[1]-LM_BOX_SIZE/2.0,1)
				if (lmcx>=(TRAJ_XLIM[0]-LM_BOX_SIZE/2.0)) and (lmcx<=(TRAJ_XLIM[1]+LM_BOX_SIZE/2.0)) and (lmcy>=(TRAJ_YLIM[0]-LM_BOX_SIZE/2.0)) and (lmcy<=(TRAJ_YLIM[1]+LM_BOX_SIZE/2.0)):
					continue
				else:
					find_lmc = False

			lmx = np.random.uniform(lmcx-LM_BOX_SIZE/2.0,lmcx+LM_BOX_SIZE/2.0,N_LM)[:,None]
			lmy = np.random.uniform(lmcy-LM_BOX_SIZE/2.0,lmcy+LM_BOX_SIZE/2.0,N_LM)[:,None]

			self.RFID = np.hstack((lmx, lmy))
			#self.RFID = np.array([[1.86912964, 14.6324081],
			#		[2.19225334, 14.72220263],
			#		[-1.17720019,  9.53217515],
			#		[3.67062618, 10.30577019],
			#		[0.70654015, 15.56967821],
			#		])

			#print(self.RFID)

			self.particles = [Particle(N_LM) for i in range(N_PARTICLE)]

			count_vis = 0
			#add visible lm to particles
			for i in range(len(self.RFID[:, 0])):
				dx = self.RFID[i, 0] - self.xTrue[0, 0]
				dy = self.RFID[i, 1] - self.xTrue[1, 0]
				d = math.sqrt(dx**2 + dy**2)
				angle = math.atan2(dy, dx) - self.xTrue[2, 0]

				delta = np.abs( np.arctan2(dy,dx) - self.pi_2_pi(self.xTrue[2,0] + self.theta) )
				if delta > np.pi:
					delta = 2*np.pi - delta

				if d <= MAX_RANGE and np.rad2deg(delta)<=(FOV/2.0):
					count_vis += 1
					dn = d + np.random.randn() * Qsim[0, 0]  # add noise
					anglen = angle + np.random.randn() * Qsim[1, 1]  # add noise
					zi = np.matrix([dn, self.pi_2_pi(anglen), i])
					for ip in range(N_PARTICLE):
						self.particles[ip] = self.add_new_lm(self.particles[ip], zi, Q)
			count_vis = 1

		#get state vector
		self.state = self.get_obs_fixedLM()
		#self.state = self.get_obs()

		return self.state

	def render_world(self):

		plt.figure(figsize=(4,4))
		plt.plot(self.RFID[:, 0], self.RFID[:, 1], "*k")

		if self.time > 0.0:
			for iz in range(len(self.z[:, 0])):
				lmid = int(self.z[iz, 2])
				plt.plot([self.xEst[0, 0], self.RFID[lmid, 0]], [self.xEst[1, 0], self.RFID[lmid, 1]], "-k")

			for i in range(N_PARTICLE):
				plt.plot(self.particles[i].x, self.particles[i].y, ".r")
				plt.plot(self.particles[i].lm[:, 0], self.particles[i].lm[:, 1], "xb")

			plt.plot(np.array(self.hxTrue[0, :]).flatten(), np.array(self.hxTrue[1, :]).flatten(), "-b")
			plt.plot(np.array(self.hxDR[0, :]).flatten(), np.array(self.hxDR[1, :]).flatten(), "-k")
			plt.plot(np.array(self.hxEst[0, :]).flatten(), np.array(self.hxEst[1, :]).flatten(), "-r")

			plt.plot(self.xEst[0], self.xEst[1], "xk")

		#plot FOV
		x0, y0 = self.xTrue.item(0), self.xTrue.item(1)
		theta1 = self.xTrue.item(2) + self.theta + np.deg2rad(FOV/2.0)
		x1 = x0 + MAX_RANGE*np.cos(theta1)
		y1 = y0 + MAX_RANGE*np.sin(theta1)
		theta2 = self.xTrue.item(2) + self.theta - np.deg2rad(FOV/2.0)
		x2 = x0 + MAX_RANGE*np.cos(theta2)
		y2 = y0 + MAX_RANGE*np.sin(theta2)

		plt.plot([x0,x1], [y0,y1], 'g-')
		plt.plot([x0,x2], [y0,y2], 'g-')

		plt.title('Time: ' + str(round(self.time, 2)))
		plt.xlim(PLT_XLIM)
		plt.ylim(PLT_YLIM)
		plt.gca().set_aspect('equal', adjustable='box')
		plt.grid(True)

		return #x0, x1, x2

	def render_obs(self):

		fig = plt.figure(figsize=(8,8))
		ax = fig.add_subplot(111, projection='polar')

		#theta = np.arange(0.0, 360.0, BIN_SIZE) + BIN_SIZE/2.0
		ptheta, prange = np.zeros((1,0)), np.zeros((1,0))
		n_bins = int(360/BIN_SIZE)
		for ibin in range(n_bins):
			if self.state[ibin]==0:
				continue
			
			for il in range(int(self.state[ibin])):
				#print(ibin, il, ptheta.shape, 
				ptheta = np.hstack(( ptheta, np.arange(ibin*BIN_SIZE, (ibin+1)*BIN_SIZE, 0.1)[None,:] ))
				prange = np.hstack(( prange, np.ones(( 1, int(BIN_SIZE/0.1) ))*(self.state[n_bins+ibin] + il*1) ))

		ax.scatter(np.deg2rad(ptheta), prange, s=1.0)
		
		sr = np.linspace(0.0, MAX_RANGE, 1000)
		st1 = np.ones(sr.shape)*(self.theta + np.deg2rad(FOV/2.0))
		st2 = np.ones(sr.shape)*(self.theta - np.deg2rad(FOV/2.0))
		st = np.hstack(( st1, st2 ))
		sr = np.hstack(( sr, sr ))
		ax.scatter(st, sr, s=10.0)

		ax.set_rmax(MAX_RANGE)
		ax.set_xticks(np.deg2rad(np.arange(0,360,BIN_SIZE)))
		ax.set_rticks(np.arange(0, MAX_RANGE, 10))
		ax.grid(True)

		return

	def render_obs_fixedLM(self):

		plt.figure(figsize=(4,4))
		
		for ilm in range(N_LM):
			plt.plot(self.state[ilm*2], self.state[ilm*2+1], "*b")

		x0, y0 = 0.0, 0.0
		theta1 = self.theta + np.deg2rad(FOV/2.0)
		x1 = x0 + MAX_RANGE*np.cos(theta1)
		y1 = y0 + MAX_RANGE*np.sin(theta1)
		theta2 = self.theta - np.deg2rad(FOV/2.0)
		x2 = x0 + MAX_RANGE*np.cos(theta2)
		y2 = y0 + MAX_RANGE*np.sin(theta2)

		plt.plot([x0,x1], [y0,y1], 'g-')
		plt.plot([x0,x2], [y0,y2], 'g-')

		plt.xlim(PLT_XLIM)
		plt.ylim(PLT_YLIM)
		plt.grid(True)
		return


	def get_obs(self):

		n_bins = int(360/BIN_SIZE)
		n_landmarks = np.ones(n_bins)*1000
		closest_dists = np.ones(n_bins)*1000
	
		lmSeen = self.particles[0].lmSeen

		for ilm in lmSeen:
			lmx, lmy = 0.0, 0.0
			for ip in range(N_PARTICLE):
				lmx += self.particles[ip].w * self.particles[ip].lm[ilm,0]
				lmy += self.particles[ip].w * self.particles[ip].lm[ilm,1]			
			lmx -= self.x_state[0]
			lmy -= self.x_state[1]

			newx = lmx*np.cos(self.x_state[2]) + lmy*np.sin(self.x_state[2])
			newy = -lmx*np.sin(self.x_state[2]) + lmy*np.cos(self.x_state[2])

			phi = np.rad2deg(np.arctan2(newy, newx))
			if phi<0:
				phi += 360.0

			bin_number = int( phi / BIN_SIZE )
			
			#add lm to bin
			n_landmarks[bin_number] += 1

			#check if lm is closest
			d_lm = np.sqrt(newx**2 + newy**2)
			if d_lm < closest_dists[bin_number]:
				closest_dists[bin_number] = d_lm

		#x_cov, y_cov, yaw_cov = self.get_particle_covariance() 
		#state = np.hstack(( n_landmarks, closest_dists, self.theta, x_cov, y_cov, yaw_cov ))

		#state = np.hstack(( n_landmarks, closest_dists, self.theta+np.deg2rad(FOV/2.0), self.theta-np.deg2rad(FOV/2.0) ))
		state = np.hstack(( n_landmarks, closest_dists, np.cos(self.theta), np.sin(self.theta) ))

		return state
		
	def get_obs_fixedLM(self):

		lm_pos = np.ones(2*N_LM)*1000
		#lm_pos = np.ones(2*N_LM)*0

		lmSeen = self.particles[0].lmSeen
		for ilm in lmSeen:
			lmx, lmy = 0.0, 0.0
			for ip in range(N_PARTICLE):
				lmx += self.particles[ip].w * self.particles[ip].lm[ilm,0]
				lmy += self.particles[ip].w * self.particles[ip].lm[ilm,1]			
			lmx -= self.x_state[0]
			lmy -= self.x_state[1]

			newx = lmx*np.cos(self.x_state[2]) + lmy*np.sin(self.x_state[2])
			newy = -lmx*np.sin(self.x_state[2]) + lmy*np.cos(self.x_state[2])

			lm_pos[ilm*2] = newx / PLT_XLIM[1]
			lm_pos[ilm*2+1] = newy / PLT_YLIM[1]

		state = np.hstack(( lm_pos, np.cos(self.theta), np.sin(self.theta) ))

		return state


	def get_reward(self):

		if self.reward_type is 'Error':

			pos_error = np.linalg.norm((self.x_state-self.xTrue)[0:2])
			yaw_error = np.abs( (self.x_state-self.xTrue)[2] )
			if yaw_error > np.pi:
				yaw_error = 2*np.pi - yaw_error
			yaw_error = np.rad2deg(yaw_error)

			r = 1 / ( pos_error + ALPHA*yaw_error + 1e-3)
			return r.item(0)/100.0

		elif self.reward_type is 'Covariance':
			
			x_cov, y_cov, yaw_cov = self.get_particle_covariance()					

			pos_cov = np.sqrt(x_cov**2 + y_cov**2)# + np.random.randn()*R[0,0]
			yaw_cov = np.var(yaw_cov)# + np.random.randn()*R[1,1]

			r = 1 / ( pos_cov + ALPHA*yaw_cov + 1e-3)
			return r/100.0
		
		return

	def get_reward_LM(self):

		r = 0.0
		for i in range(len(self.RFID[:, 0])):
			dx = self.RFID[i, 0] - self.xTrue[0, 0]
			dy = self.RFID[i, 1] - self.xTrue[1, 0]
			d = math.sqrt(dx**2 + dy**2)
			angle = math.atan2(dy, dx) - self.xTrue[2, 0]

			delta = np.abs( np.arctan2(dy,dx) - self.pi_2_pi(self.xTrue[2,0] + self.theta) )
			if delta > np.pi:
				delta = 2*np.pi - delta

			if d <= MAX_RANGE and np.rad2deg(delta)<=(FOV/2.0):
				r += 1.0	
		return r

	def get_particle_covariance(self):
		
		x_, y_, yaw_ = [], [], []
		for p in self.particles:
			x_.append(p.x)
			y_.append(p.y)
			yaw_.append(p.yaw)

		return np.var(x_), np.var(y_), np.var(yaw_)

	def fast_slam2(self, particles, u, z):
	    
		particles = self.predict_particles(particles, u)

		particles = self.update_with_observation(particles, z)

		particles = self.resampling(particles)
	    
		return particles


	def normalize_weight(self, particles):

		sumw = sum([p.w for p in particles])

		try:
			for i in range(N_PARTICLE):
				particles[i].w /= sumw
		except ZeroDivisionError:
			for i in range(N_PARTICLE):
				particles[i].w = 1.0 / N_PARTICLE

			return particles

		return particles


	def calc_final_state(self, particles):

		xEst = np.zeros((STATE_SIZE, 1))

		particles = self.normalize_weight(particles)

		for i in range(N_PARTICLE):
			xEst[0, 0] += particles[i].w * particles[i].x
			xEst[1, 0] += particles[i].w * particles[i].y
			xEst[2, 0] += particles[i].w * particles[i].yaw

		xEst[2, 0] = self.pi_2_pi(xEst[2, 0])

		return xEst


	def predict_particles(self, particles, u):

		p_out = particles
		for i in range(N_PARTICLE):
			px = np.zeros((STATE_SIZE, 1))
			px[0, 0] = particles[i].x
			px[1, 0] = particles[i].y
			px[2, 0] = particles[i].yaw
			ud = u + (np.matrix(np.random.randn(1, 2)) * R).T  # add noise
			px = self.motion_model(px, ud)
			p_out[i].x = px[0, 0]
			p_out[i].y = px[1, 0]
			p_out[i].yaw = px[2, 0]

		return p_out


	def add_new_lm(self, particle, z, Q):

		r = z[0, 0]
		b = z[0, 1]
		lm_id = int(z[0, 2])

		s = math.sin(self.pi_2_pi(particle.yaw + b))
		c = math.cos(self.pi_2_pi(particle.yaw + b))

		#particle.lm[lm_id, 0] = particle.x + r * c
		#particle.lm[lm_id, 1] = particle.y + r * s

		#remove later
		particle.lm[lm_id, 0] = self.RFID[lm_id, 0]
		particle.lm[lm_id, 1] = self.RFID[lm_id, 1]

		# covariance
		Gz = np.matrix([[c, -r * s],
				[s, r * c]])

		particle.lmP[2 * lm_id:2 * lm_id + 2] = 0 * Gz * Q * Gz.T

		particle.lmSeen.append(lm_id)

		return particle


	def compute_jacobians(self, particle, xf, Pf, Q):
		dx = xf[0, 0] - particle.x
		dy = xf[1, 0] - particle.y
		d2 = dx**2 + dy**2
		d = math.sqrt(d2)

		zp = np.matrix([[d, self.pi_2_pi(math.atan2(dy, dx) - particle.yaw)]]).T

		Hv = np.matrix([[-dx / d, -dy / d, 0.0],
				[dy / d2, -dx / d2, -1.0]])

		Hf = np.matrix([[dx / d, dy / d],
				[-dy / d2, dx / d2]])

		Sf = Hf * Pf * Hf.T + Q

		return zp, Hv, Hf, Sf


	def update_KF_with_cholesky(self, xf, Pf, v, Q, Hf):
		PHt = Pf * Hf.T
		S = Hf * PHt + Q

		S = (S + S.T) * 0.5
		SChol = np.linalg.cholesky(S).T
		SCholInv = np.linalg.inv(SChol)
		W1 = PHt * SCholInv
		W = W1 * SCholInv.T

		x = xf + W * v
		P = Pf - W1 * W1.T

		return x, P


	def update_landmark(self, particle, z, Q):

		lm_id = int(z[0, 2])
		xf = np.matrix(particle.lm[lm_id, :]).T
		Pf = np.matrix(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

		zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q)

		dz = z[0, 0: 2].T - zp
		dz[1, 0] = self.pi_2_pi(dz[1, 0])

		xf, Pf = self.update_KF_with_cholesky(xf, Pf, dz, Q, Hf)

		particle.lm[lm_id, :] = xf.T
		particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

		return particle


	def compute_weight(self, particle, z, Q):

		lm_id = int(z[0, 2])
		xf = np.matrix(particle.lm[lm_id, :]).T
		Pf = np.matrix(particle.lmP[2 * lm_id:2 * lm_id + 2])
		zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q)

		dz = z[0, 0: 2].T - zp
		dz[1, 0] = self.pi_2_pi(dz[1, 0])

		S = particle.lmP[2 * lm_id:2 * lm_id + 2]

		try:
			invS = np.linalg.inv(S)
		except np.linalg.linalg.LinAlgError:
			#add later
			#print("Singular!")
			return 1.0

		num = math.exp(-0.5 * dz.T * invS * dz)
		den = 2.0 * math.pi * math.sqrt(np.linalg.det(S))

		w = num / den

		return w


	def proposal_sampling(self, particle, z, Q):

		lm_id = int(z[0, 2])
		xf = np.matrix(particle.lm[lm_id, :]).T
		Pf = np.matrix(particle.lmP[2 * lm_id:2 * lm_id + 2])
		# State
		x = np.matrix([[particle.x, particle.y, particle.yaw]]).T
		P = particle.P
		zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q)

		Sfi = np.linalg.inv(Sf)
		dz = z[0, 0: 2].T - zp
		dz[1, 0] = self.pi_2_pi(dz[1, 0])

		Pi = np.linalg.inv(P)

		particle.P = np.linalg.inv(Hv.T * Sfi * Hv + Pi)  # proposal covariance
		x += particle.P * Hv.T * Sfi * dz  # proposal mean

		particle.x = x[0, 0]
		particle.y = x[1, 0]
		particle.yaw = x[2, 0]

		return particle


	def update_with_observation(self, particles, z):

		for iz in range(len(z[:, 0])):

			lmid = int(z[iz, 2])

			for ip in range(N_PARTICLE):
				# new landmark
				if particles[ip].lm[lmid, 0] == 0:
					particles[ip] = self.add_new_lm(particles[ip], z[iz, :], Q)
				# known landmark
				else:
					w = self.compute_weight(particles[ip], z[iz, :], Q)
					particles[ip].w *= w

					particles[ip] = self.update_landmark(particles[ip], z[iz, :], Q)
					particles[ip] = self.proposal_sampling(particles[ip], z[iz, :], Q)

		return particles


	def resampling(self, particles):
		
		particles = self.normalize_weight(particles)

		pw = []
		for i in range(N_PARTICLE):
			pw.append(particles[i].w)

		pw = np.matrix(pw)

		Neff = 1.0 / (pw * pw.T)[0, 0]  # Effective particle number
	    
		if Neff < NTH:  # resampling
		
			wcum = np.cumsum(pw)
			base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
			resampleid = base + np.random.rand(base.shape[1]) / N_PARTICLE
		
			inds = []
			ind = 0
			for ip in range(N_PARTICLE):
				while ((ind < wcum.shape[1] - 1) and (resampleid[0, ip] > wcum[0, ind])):
					ind += 1
				inds.append(ind)

			tparticles = particles[:]
			for i in range(len(inds)):
				particles[i].x = tparticles[inds[i]].x
				particles[i].y = tparticles[inds[i]].y
				particles[i].yaw = tparticles[inds[i]].yaw
				particles[i].lm = tparticles[inds[i]].lm[:, :]
				particles[i].lmP = tparticles[inds[i]].lmP[:, :]
				particles[i].w = 1.0 / N_PARTICLE

		return particles


	def calc_input(self, time):

		dt = time - TIME_OFFSET

		v = np.sqrt(1 + (TRAJ_AMP*TRAJ_FREQ*np.cos(TRAJ_FREQ*dt))**2)
		yawrate = -TRAJ_AMP*TRAJ_FREQ*TRAJ_FREQ*np.sin(TRAJ_FREQ*dt)

		u = np.matrix([v, yawrate]).T

		#stay at starting point for TIME_OFFSET seconds
		if dt < 0:
			u *= 0

		return u


	def observation(self, xTrue, xd, u, RFID):

		xTrue = self.motion_model(xTrue, u)
		z = np.matrix(np.zeros((0, 3)))

		for i in range(len(RFID[:, 0])):

			dx = RFID[i, 0] - xTrue[0, 0]
			dy = RFID[i, 1] - xTrue[1, 0]
			d = math.sqrt(dx**2 + dy**2)
			angle = math.atan2(dy, dx) - xTrue[2, 0]

			delta = np.abs( np.arctan2(dy,dx) - self.pi_2_pi(xTrue[2,0] + self.theta) )
			if delta > np.pi:
				delta = 2*np.pi - delta

			if d <= MAX_RANGE and np.rad2deg(delta)<=(FOV/2.0):
				dn = d + np.random.randn() * Qsim[0, 0]  # add noise
				anglen = angle + np.random.randn() * Qsim[1, 1]  # add noise
				zi = np.matrix([dn, self.pi_2_pi(anglen), i])
				z = np.vstack((z, zi))

		# add noise to input
		ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
		ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1] + OFFSET_YAWRATE_NOISE
		ud = np.matrix([ud1, ud2]).T

		xd = self.motion_model(xd, ud)

		return xTrue, z, xd, ud


	def motion_model(self, x, u):

		F = np.matrix([[1.0, 0, 0],
				[0, 1.0, 0],
				[0, 0, 1.0]])

		B = np.matrix([[self.DT * math.cos(x[2, 0]), 0],
				[self.DT * math.sin(x[2, 0]), 0],
				[0.0, self.DT]])

		x = F * x + B * u

		x[2, 0] = self.pi_2_pi(x[2, 0])

		return x


	def pi_2_pi(self, angle):
		return (angle + math.pi) % (2 * math.pi) - math.pi
