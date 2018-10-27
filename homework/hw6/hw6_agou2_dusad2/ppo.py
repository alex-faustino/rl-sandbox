#import Policy from policy_gradient
from tqdm import tqdm_notebook
import numpy as np 
import torch
import torch.nn.functional as F
import torch.optim as optim

class Batch:
    def __init__(self, num_traj, num_step):
        self.states = torch.zeros((num_traj, num_step, 3))
        self.actions = torch.zeros((num_traj, num_step))
        self.rewards = torch.zeros((num_traj, num_step))
        self.logprobs = torch.zeros((num_traj, num_step))
        self.values = torch.zeros((num_traj, num_step))
        self.num_traj = num_traj
        self.num_step = num_step


    def add_step(self, traj, step, state, action, reward, logprob, value):
        self.states[traj, step] = state
        self.actions[traj, step] = action
        self.rewards[traj, step] = reward
        self.logprobs[traj, step] = logprob
        self.values[traj, step] = value

    def __iter__(self):
        for i in range(self.num_traj):
            batch = (self.states[i], self.actions[i], self.rewards[i], self.logprobs[i], self.values[i])
            yield batch 

class PPO:
    def __init__(self, env, policy):
        self.policy = policy
        self.env = env
        

    def get_batch(self,  num_traj, num_step):
        batch = Batch(num_traj, num_step)
        state = torch.from_numpy(self.env.reset()).float()
        for traj in range(num_traj):
            for step in range(num_step):
                action, logprob, value = self.policy.sample(state)
                next_state, reward, done, _ = self.env.step((action.item(),))
                batch.add_step(traj, step, state, action, torch.tensor(reward).float(), logprob, value)
                state = torch.from_numpy(next_state).float()
        return batch
    
    def get_advantages(self, traj, gamma):
        returns = torch.zeros((self.num_step))
        states, actions, rewards, logprobs, values = traj 
        # rewards = (rewards - rewards.mean())/(rewards.std() + 1e-5)
        returns[-1] = rewards[-1] + gamma*values[-1]
        for step in range(self.num_step -2, -1, -1):
            returns[step] = rewards[step] + returns[step+1]*gamma
        advantages = returns - values 
        return advantages, returns
        
    
    def train(self, num_episode=10, num_step=20, num_traj=5,alpha=0.1, gamma=0.7, num_epochs=5, epsilon=1e-5, c=1):
        self.num_step = num_step
        optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        losses = torch.zeros((num_episode, num_epochs))
        rewards = torch.zeros(num_episode)
        for episode in tqdm_notebook(range(num_episode)):
            with torch.no_grad():
                batch = self.get_batch(num_traj, num_step)
            rewards[episode] = batch.rewards.sum()
            for epoch in range(num_epochs):
                for traj in batch:
                    loss = self.loss(traj, gamma, epsilon, c)
                    optimizer.zero_grad()
                    loss.backward()
                    losses[episode, epoch] = loss.item()
                    optimizer.step()

        return rewards, losses
    
    def loss(self, traj, gamma, epsilon, c):
        advantages, returns = self.get_advantages(traj, gamma)
        old_states, old_actions, old_rewards, old_logprobs, old_values = traj

        logprobs = torch.zeros(old_logprobs.shape)
        values = torch.zeros(old_logprobs.shape)
        for i, (old_state, old_action) in enumerate(zip(old_states, old_actions)):
            logprobs[i], values[i] = self.policy.eval(old_state, old_action)

        ratios = torch.exp(logprobs - old_logprobs)
        L = ratios*advantages
        L_CLIP_CLAMPED = torch.clamp(ratios, 1-epsilon, 1+epsilon)*advantages
        L_CLIP = torch.min(L, L_CLIP_CLAMPED).mean()

        L_VF = 0.5*F.mse_loss(returns, values) # values = V_target, returns = V_theta_S

        loss = -L_CLIP + c*L_VF
        # print("L: {} \n ratios: {} \n advantages: {} \n logprobs: {} \n old_logprobs: {}".format(L, ratios, advantages, logprobs, old_logprobs))
        return loss

    def test(self, num_step=1000):
        state = torch.from_numpy(self.env.reset()).float()
        for step in range(num_step):
            action, logprob, value = self.policy.sample(state)
            next_state, reward, done, _ = self.env.step((action.item(),))
            self.env.render()
            state = torch.from_numpy(next_state).float()


