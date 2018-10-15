#import Policy from policy_gradient
from tqdm import tqdm_notebook
import numpy as np 

class Memory:
    def __init__(self, num_episode, num_traj, num_step, action_space, state_space):
        self.num_episode = num_episode
        self.num_traj = num_traj
        self.num_step = num_step
        self.action_space = action_space
        self.state_space = state_space

        self.states = np.zeros((num_episode, num_traj, num_step), dtype=int)
        self.rewards =  np.zeros((num_episode, num_traj, num_step))
        self.gradients = np.zeros((num_episode, num_traj, num_step, self.action_space))
        self.probs = np.zeros((num_episode, num_traj, num_step, self.action_space))
        self.actions = np.zeros((num_episode, num_traj, num_step), dtype=int)

    def sample(self, number_of_samples, episode_limit):
        episode_indices = np.random.choice(episode_limit, size=number_of_samples)
        trajectory_indices = np.random.choice(self.num_traj, size=number_of_samples)
        return zip(episode_indices, trajectory_indices)

    def add_step(self, ep, traj, step, probs, action, gradient, state, reward):
        self.probs[ep, traj, step] = probs
        self.actions[ep, traj, step] = action
        self.gradients[ep, traj, step] =  gradient
        self.states[ep, traj, step] = state
        self.rewards[ep, traj, step] = reward

class Reinforce:

    def get_update(self, ep, traj, causality=False):
        if causality:
            causality_update = np.zeros((self.state_space,self.action_space))
            reward_vec = np.zeros(self.memory.num_step)
            for step in range(self.memory.num_step):
                reward_vec[:step+1] += self.memory.rewards[ep,traj,step] 
            for step in range(self.memory.num_step):
                state = self.memory.states[ep, traj, step]
                causality_update[state] += self.memory.gradients[ep, traj, step]*reward_vec[step]
            return causality_update
        else:
            gradient_estimate = np.zeros((self.state_space,self.action_space))
            for step in range(self.memory.num_step):
                state = self.memory.states[ep, traj, step]
                gradient_estimate[state] += self.memory.gradients[ep, traj, step]
            return self.memory.rewards[ep, traj].sum()*gradient_estimate

    def __init__(self, env, state_space, action_space, policy, importance_sampling=False):
        self.state_space = state_space
        self.action_space = action_space
        self.policy = policy
        self.env = env
        
    
    def train(self, num_episode=1000, num_step=20, num_traj=5,alpha=0.1, gamma=1, 
              importance_sampling=False, num_samples=None, causality=False
             ):
        self.memory = Memory(num_episode, num_traj, num_step, self.action_space, self.state_space)
        for ep in tqdm_notebook(range(num_episode)):
            state = self.env.reset()
            for traj in range(num_traj):
                for step in range(num_step):
                    action, probs = self.policy.get(state)
                    gradient = self.policy.gradient(probs, action)
                    next_state, reward, done, _ = self.env.step(action)
                    self.memory.add_step(ep, traj, step, probs, action, gradient, state, reward)
                    state = next_state
            
            if importance_sampling:
                updates = np.zeros((self.state_space, self.action_space,num_samples))
                for ep, traj in self.memory.sample(num_samples, ep+1):
                    lp_d , lp_n = 0, 0 
                    for step in range(num_step):
                        _, probs = self.policy.get(state)
                        lp_d += np.log(probs[self.memory.actions[ep, traj, step]])
                        lp_n += np.log(self.memory.probs[ep,traj, step, self.memory.actions[ep, traj, step]])
                    if (np.exp(lp_n - lp_d)) > 100:
                        continue
                    updates[:,:,traj] = (np.exp(lp_n - lp_d))*self.get_update(ep, traj, causality)
            else:
                updates = np.zeros((self.state_space, self.action_space,num_traj))
                for traj in range(num_traj):
                    updates[:,:,traj] = self.get_update(ep, traj, causality)
            self.policy.theta += alpha*np.mean(updates, -1)
        return self.memory.rewards.sum(axis=(1, 2)), self.memory.rewards
        
    def test(self, num_step, render=True):
        state = self.env.reset()
        trajectory = np.zeros(num_step)
        actions = np.zeros(num_step)
        rewards = np.zeros(num_step)
        
        imgs = []
        for step in range(num_step):
            action, probs = self.policy.get(state)
            if render:
                imgs.append(self.env.render(mode='rgb_array'))
            next_state, reward, done, _ = self.env.step(action)
            
            trajectory[step] = state
            actions[step] = action
            rewards[step] = reward
            
            state= next_state
        return rewards, trajectory, actions, imgs