import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple

torch.manual_seed(0)

### linear functions to try
### constant/simple baseline functions to start with 

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.hidden = nn.Linear(3, 100)
        self.mu = nn.Linear(100, 1)
        self.sigma = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        mu = 2.0 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        ### Output: mean and variance from which the continuous action is sampled
        return (mu, sigma)  ## Gaussian policy


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.hidden = nn.Linear(3, 100)
        self.value = nn.Linear(100, 1) 

    def forward(self, x):
        x = F.relu(self.hidden(x))
        value_func = self.value(x)
        return value_func ### Output: value function of the state obtained from critic
    
class Agent():

    def __init__(self, env):
        self.env = env
        self.training_step = 0
        self.actorObj = Actor().float()
        self.criticObj = Critic().float()
        self.buffer = []
        self.counter = 0
        #self.reset()
                
    def select_action(self, state):
        ### Convert the state into a tensor which is later given as input to the "actor deep neural network"
        ### To obtain mu and sigma as output
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.actorObj(state)
        ### Normal distribution considered based on mu and sigma computed
        Gpdf = Normal(mu, sigma)
        action = Gpdf.sample() ### in tensor form
        lprob = Gpdf.log_prob(action) ### in tensor form
        action.clamp(-2.0, 2.0)
        return action.item(), lprob.item() ### convert to number with item()

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.mem_size == 0

    def update(self):
        self.training_step += 1

        store_s = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        store_a = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        store_r = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        store_snew = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        store_lprob = torch.tensor([t.lprob for t in self.buffer], dtype=torch.float).view(-1, 1)

        rnorm = (store_r - store_r.mean()) / (store_r.std() + 1e-5)
        with torch.no_grad():
            value_func = rnorm + self.gamma * self.criticObj(store_snew)

        Ahat = (value_func - self.criticObj(store_s)).detach() ### advantage function

        for pp in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.mem_size)), self.batch_size, False):
                (mu, sigma) = self.actorObj(store_s[index])
                Gpdf = Normal(mu, sigma)
                lprob = Gpdf.log_prob(store_a[index])
                pratio = torch.exp(lprob - store_lprob[index])

                Lclip_f = pratio * Ahat[index]
                Lclip_s = torch.clamp(pratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * Ahat[index]
                Lclip = -torch.min(Lclip_f, Lclip_s).mean()

                self.opt_actor.zero_grad()
                Lclip.backward()
                nn.utils.clip_grad_norm_(self.actorObj.parameters(), self.max_grad)
                self.opt_actor.step()

                Lvf = F.smooth_l1_loss(self.criticObj(store_s[index]), value_func[index])
                self.opt_critic.zero_grad()
                Lvf.backward()
                nn.utils.clip_grad_norm_(self.criticObj.parameters(), self.max_grad)
                self.opt_critic.step()

        del self.buffer[:]

    def train_model(self, param): 
        self.clip_param = param['clip']
        self.max_grad = param['max_grad']
        self.ppo_epoch = param['ppo_epoch']
        self.mem_size = param['mem_size']
        self.batch_size = param['batch_size']
        self.gamma = param['gamma'] 
        self.no_eps = param['no_eps']
        self.eps_len = param['eps_len']
        self.lr_actor = param['lr']
        self.lr_critic = param['lr']
        
        MemoryBuffer = namedtuple('MemoryBuffer', ['state', 'action', 'lprob', 'reward', 'next_state'])
        self.opt_actor = optim.Adam(self.actorObj.parameters(), lr=self.lr_actor)
        self.opt_critic = optim.Adam(self.criticObj.parameters(), lr=self.lr_actor)
        
        ### Loop over the number of episodes
        store_avg_rewards = []
        for i_ep in range(self.no_eps):
            reward_eps = 0
            ### First step is to reset the states
            state = self.env.reset()
            ### Loop over the episode length for each trajectory/episode
            for t in range(self.eps_len):
                ### Compute the action and obtain the action as well as the log prob
                action, lprob = self.select_action(state)
            
                ### Get feedback from environment about reward and state
                next_state, reward, done, info = self.env.step([action])
        
                ### Render the animation!!
                #self.env.render()
                print('reward: ', reward)
            
                ### Update the storage of agent 
                if self.store(MemoryBuffer(state, action, lprob, (reward + 8) / 8, next_state)):
                    self.update()
                reward_eps += reward
                state = next_state

            ### score => total rewards at the end of each episode length
            store_avg_rewards.append([i_ep, reward_eps])           
            
        #self.env.close()
        
        return store_avg_rewards

    def test_model(self, eps_len): 
        
        ### Loop over the number of episodes
        record_pos = []
        record_action = []
        reward_eps = 0
        
        ### First step is to reset the states
        state = self.env.reset()
 
        ### Loop over the episode length for each trajectory/episode
        for t in range(eps_len):
            record_pos.append(state)

            ### Compute the action and obtain the action as well as the log prob
            action, lprob = self.select_action(state)
            record_action.append(action)
            
            ### Get feedback from environment about reward and state
            next_state, reward, done, info = self.env.step([action])
        
            ### Render the animation!!
            self.env.render()
            
            reward_eps += reward
            state = next_state
            
        self.env.close()
        
        return record_pos, record_action, reward_eps