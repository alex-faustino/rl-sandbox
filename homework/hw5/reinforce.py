#import Policy from policy_gradient
from tqdm import tqdm_notebook
import numpy as np 

class Reinforce:
    def __init__(self, env, state_space, action_space, policy, importance_sampling=False):
        self.state_space = state_space
        self.action_space = action_space
        self.policy = policy
        self.env = env
        
    
    def train(self, episode_num=1000, traj_length=20, num_traj=5,alpha=0.1, gamma=1, 
              importance_sampling=False, num_samples=None, causality=False
             ):
        self.importance_sampling = importance_sampling
        self.trajectories = np.zeros((episode_num, num_traj, traj_length), dtype=int)
        self.rewards =  np.zeros((episode_num, num_traj, traj_length))
        self.gradients = np.zeros((episode_num, num_traj, traj_length, self.action_space))
        self.logprobs = np.zeros((episode_num, num_traj, traj_length, self.action_space))
        self.actions = np.zeros((episode_num, num_traj, traj_length), dtype=int)
        
        for ep in tqdm_notebook(range(episode_num)):
            state = self.env.reset()
            updates = np.zeros((self.state_space, self.action_space,num_traj)) if importance_sampling is False else np.zeros((self.state_space, self.action_space,num_samples))
            for traj in range(num_traj):
                for step in range(traj_length):
                    action, probs = self.policy.get(state)
                    self.logprobs[ep, traj, step] = np.log(probs) 
                    self.actions[ep, traj, step] = action
                    self.gradients[ep, traj, step] =  self.policy.gradient(probs, action)
                    self.trajectories[ep, traj, step] = state
                    state, reward, done, _ = self.env.step(action)
                    self.rewards[ep,traj,step] = reward
                    if causality:
                        self.rewards[ep,traj,:step] += reward
                    
                if not self.importance_sampling:
                    # updates[state,:,traj] += self.gradients[ep, traj].sum(axis=0)*self.rewards[ep, traj].sum()
                    gradient_estimate = np.zeros((self.state_space,self.action_space))
                    for step in range(traj_length):
                        state = self.trajectories[ep, traj, step]
                        if causality:
                            gradient_estimate[state] += self.gradients[ep, traj, step]*self.rewards[ep,traj, step]
                        else:
                            gradient_estimate[state] += self.gradients[ep, traj, step]
                    if causality:
                        updates[:,:,traj] = gradient_estimate
                    else:
                        updates[:,:,traj] = self.rewards[ep, traj].sum()*gradient_estimate
#             if self.importance_sampling:
#                 episode_indices, trajectory_indices = self.sample_trajectories(num_samples, ep+1, num_traj) #2D
#                 # print(episode_indices)
#                 # episode_indices = np.full(num_samples, ep)
#                 sampled_states = self.trajectories[episode_indices, trajectory_indices] # 
#                 sampled_rewards = self.rewards[episode_indices, trajectory_indices]
#                 sampled_grads = self.gradients[episode_indices, trajectory_indices]
#                 for traj in range(num_samples):
#                     likelihood_ratio = 0
#                     for step in range(traj_length):
#                         state = sampled_states[traj, step]
#                         gradient_estimate = np.zeros((self.state_space,self.action_space))
#                         gradient_estimate[state] += sampled_grads[traj,step]
#                         _, probs = self.policy.get(state)
                        
#                         likelihood_ratio += self.logprobs[ep, traj, step, self.actions[ep, traj, step]]
#                         likelihood_ratio -= np.log(probs)[self.actions[ep, traj, step]]
                        
#                     updates[:,:,traj] = np.exp(likelihood_ratio)*sampled_rewards[traj].sum()*gradient_estimate

            self.policy.theta += alpha*np.mean(updates, -1)
        return self.rewards.sum(axis=(1, 2)) if not causality else self.rewards[:,:,0].sum(axis=1)
    
    def sample_trajectories(self, number_of_samples, episode_limit, num_traj):

        episode_indices = np.random.choice(episode_limit, size=number_of_samples)
        trajectory_indices = np.random.choice(num_traj, size=number_of_samples)
        return episode_indices, trajectory_indices
    
    def test(self, num_steps, render=True):
        state = self.env.reset()
        trajectory = np.zeros(num_steps)
        actions = np.zeros(num_steps)
        rewards = np.zeros(num_steps)
        
        imgs = []
        for step in range(num_steps):
            action, probs = self.policy.get(state)
            if render:
                imgs.append(self.env.render(mode='rgb_array'))
            next_state, reward, done, _ = self.env.step(action)
            
            trajectory[step] = state
            actions[step] = action
            rewards[step] = reward
            
            state= next_state
        return rewards, trajectory, actions, imgs