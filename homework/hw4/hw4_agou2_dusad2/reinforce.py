#import Policy from policy_gradient
from tqdm import tqdm_notebook
import numpy as np 

class Reinforce:
    def __init__(self, env, state_space, action_space, policy):
        self.state_space = state_space
        self.action_space = action_space
        self.policy = policy
        self.env = env
    
    def train(self, episode_num=1000, traj_length=20, num_traj=5,alpha=0.1, gamma=1):
        ep_rewards = np.zeros(episode_num)
        for ep in tqdm_notebook(range(episode_num)):
            state = self.env.reset()
            traj_rewards = np.zeros(num_traj)
            updates = np.zeros((self.state_space, self.action_space,num_traj))
            for traj in range(num_traj):
                states = np.zeros(traj_length)
                actions = np.zeros(traj_length)
                rewards = np.zeros(traj_length)
                grads = np.zeros((traj_length, 4))
                for step in range(traj_length):
                    action, probs = self.policy.get(state)
                    grads[step] =  self.policy.gradient(probs, action)
                    states[step] = state
                    actions[step] = action 
                    next_state, reward, done, _ = self.env.step(action)
                    rewards[step] = reward
                    state = next_state
                updates[state,:,traj] = grads.sum(axis=0)*rewards.sum()
                traj_rewards[traj] = rewards.sum()
                #traj_rewards[traj] = np.sum(np.array([(gamma**i)*rewards[i] for i in range(len(rewards))]))
            self.policy.theta += alpha*np.mean(updates, -1)
            ep_rewards[ep] = traj_rewards.sum()
        return ep_rewards
    
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