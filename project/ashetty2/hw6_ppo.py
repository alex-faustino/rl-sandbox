import torch, torch.utils.data
import random, collections, math, time, datetime, os
import numpy as np
from multiprocessing import Pool

class Net(torch.nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(Net, self).__init__()
        self.V_fc1 = torch.nn.Linear(observation_dim, 100).double()
        self.V_fc2 = torch.nn.Linear(100, 100).double()
        self.V_fc3 = torch.nn.Linear(100, 100).double()
        self.V_fc4 = torch.nn.Linear(100, 1).double()
        self.mu_fc4 = torch.nn.Linear(100, action_dim).double()
        self.std_fc4 = torch.nn.Linear(100, action_dim).double()

    def forward(self, x):
        """
        Takes observation vector x and returns a scalar V and vector mu.
        x: state observation
        V: scalar value function
        mu: mean of action distribution
        """
        x = torch.tanh(self.V_fc1(x))
        x = torch.tanh(self.V_fc2(x))
        x = torch.tanh(self.V_fc3(x))
        #x = torch.leakyrelu(self.V_fc4(x))
        V = self.V_fc4(x)
        mu = self.mu_fc4(x)
        std = self.std_fc4(x)
        std = torch.sigmoid(std) # adding a small number to std may improve robustness
        return (V, mu, std)

class PPOAgent(object):
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.net = Net(self.env.observation_dim, self.env.action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def save_checkpoint(self, sm, br, r, l, lc, lv, le, s, ts, to):
        
        PATH = 'saved_models/' + sm + '.pth'
        
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'best_reward': br,
            'rewards': r,
            'losses': l,
            'losses_clip': lc,
            'losses_V': lv,
            'losses_entropy': le,
            'stds': s,
            'times_sample': ts,
            'times_opt': to
            }, PATH)

        print('==== model ' + sm + ' saved ====')
        
    def action_greedy(self, s):
        with torch.no_grad():
            (V, mu, std) = self.net(torch.from_numpy(s))
            return mu.numpy()

    def value(self, s):
        with torch.no_grad():
            (V, mu, std) = self.net(torch.from_numpy(s))
            return V.item()

    def _run_actor_for_training(self, net, env, horizon, gamma, lamb):
        with torch.no_grad():
            s = np.zeros((horizon+1, env.observation_dim))
            a = np.zeros((horizon+1, env.action_dim))
            r = np.zeros(horizon+1)
            log_pi = np.zeros(horizon+1)
            V = np.zeros(horizon+1)
          
            s_next = env.state
            for t in range(horizon+1):
                s[t,:] = s_next
                (V[t], mu_t, std_t) = net(torch.from_numpy(s[t]))
                dist = torch.distributions.normal.Normal(mu_t, std_t)
                a[t] = dist.sample().numpy()
                log_pi[t] = dist.log_prob(a[t]).sum()
                (s_next, r[t], done, info) = env.step(a[t])

            delta = r[:-1] + gamma * V[1:] - V[:-1]
            V_targ = delta

            A = np.zeros(horizon+1)
            A[-1] = 0
            for t in reversed(range(horizon)):
                A[t] = delta[t] + gamma * lamb * A[t+1]

            return {
                's': torch.from_numpy(s[:-1]),
                'a': torch.from_numpy(a[:-1]),
                'r': torch.from_numpy(r[:-1]),
                'log_pi': torch.from_numpy(log_pi[:-1]),
                'V_targ': torch.from_numpy(V_targ),
                'A': torch.from_numpy(A[:-1]),
            }

    def L_clip(self, ratio, A, epsilon):
        return torch.min(A * ratio, A * torch.clamp(ratio, 1 - epsilon, 1 + epsilon))

    def _seed_worker(self, t):
        pid = os.getpid()
        random.seed(pid + t)
        np.random.seed(pid + t)
        torch.manual_seed(pid + t)

    def train(self, save_model, load_model, load_vars, gamma, lamb, number_of_actors, number_of_iterations, horizon, number_of_epochs, minibatch_size, logstd_initial, logstd_final, epsilon, use_multiprocess=True):
        if use_multiprocess:
            number_of_workers = os.cpu_count()
            pool_of_workers = Pool(number_of_workers, self._seed_worker, (int(time.time()),))

        envs = []
        for actor in range(number_of_actors):
            # Make sure this is a deep copy
            env = self.env.copy()
            env.reset()
            envs.append(env)

        best_reward = 0.0
        rewards = []
        losses = []
        losses_clip = []
        losses_V = []
        losses_entropy = []
        stds = []
        times_sample = []
        times_opt = []

        if load_model is not '':
            PATH = 'saved_models/' + load_model + '_best.pth'
            checkpoint = torch.load(PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.net.train()
            
            if load_vars is True:
                best_reward = checkpoint['best_reward']
                rewards = checkpoint['rewards']
                losses = checkpoint['losses']
                losses_clip = checkpoint['losses_clip']
                losses_V = checkpoint['losses_V']
                losses_entropy = checkpoint['losses_entropy']
                stds = checkpoint['stds']
                times_sample = checkpoint['times_sample']
                times_opt = checkpoint['times_opt']
                
            print('==== model ' + load_model + ' loaded ====')
                
        if save_model is '':
            model_logname = 'temp'
        else:
            model_logname = save_model + '_latest'
                
        for iter in range(number_of_iterations):

            start_time = time.time()
            if use_multiprocess:
                datasets = pool_of_workers.starmap(
                    self._run_actor_for_training,
                    [(self.net, envs[actor], horizon, gamma, lamb) for actor in range(number_of_actors)]
                )
            else:
                datasets = []
                for actor in range(number_of_actors):
                    datasets.append(self._run_actor_for_training(self.net, envs[actor], horizon, gamma, lamb))
                    
            with torch.no_grad():
                data = {k: torch.cat([d[k] for d in datasets]) for k in datasets[0].keys()}

                tmp_r = torch.mean(data['r']).item()
                print(iter, tmp_r)
                rewards.append(tmp_r)

                if (save_model is not '') and (tmp_r>best_reward):
                    self.save_checkpoint(save_model + '_best', tmp_r, rewards, losses, losses_clip, losses_V, losses_entropy, stds, times_sample, times_opt)
                    best_reward = tmp_r
                
            end_time = time.time()
            times_sample.append(end_time - start_time)

            # Optimization ##################################################

            start_time = time.time()
            iter_losses = []
            iter_losses_clip = []
            iter_losses_V = []
            iter_losses_entropy = []
            iter_stds = []
            dataset = torch.utils.data.TensorDataset(data['s'], data['a'], data['log_pi'], data['V_targ'], data['A'])
            for epoch in range(number_of_epochs):
                self.optimizer.zero_grad()
                
                # Sample a subset of the data (minibatch)
                sampled_indexes = random.sample(range(len(dataset)), minibatch_size)
                (s, a, old_log_pi, V_targ, A) = dataset[sampled_indexes]
                (V, mu, std) = self.net(s)
                dist = torch.distributions.normal.Normal(mu, std)
                log_pi = dist.log_prob(a).sum(dim=1)
                ratio = torch.exp(log_pi - old_log_pi)
                loss_clip = -self.L_clip(ratio, A, epsilon).mean()
                loss_V = torch.nn.MSELoss()(V[:,0], V_targ)
                loss_entropy = -dist.entropy().sum(dim=1).mean() # torch.zeros(1, dtype=torch.float64) # log_pi.mean()
                loss = loss_clip + loss_V + loss_entropy # could add weights c_1 and c_2

                iter_losses.append(loss.item())
                iter_losses_clip.append(loss_clip.item())
                iter_losses_V.append(loss_V.item())
                iter_losses_entropy.append(loss_entropy.item())
                iter_stds.append(std.mean().item())

                loss.backward()
                self.optimizer.step()

            losses.append(np.mean(iter_losses))
            losses_clip.append(np.mean(iter_losses_clip))
            losses_V.append(np.mean(iter_losses_V))
            losses_entropy.append(np.mean(iter_losses_entropy))
            stds.append(np.mean(iter_stds))

            end_time = time.time()
            times_opt.append(end_time - start_time)
            
            # Save latest ##################################################
            self.save_checkpoint(model_logname, best_reward, rewards, losses, losses_clip, losses_V, losses_entropy, stds, times_sample, times_opt)
                
        if use_multiprocess:
            pool_of_workers.close()

        return {
            'rewards': rewards,
            'losses': losses,
            'losses_clip': losses_clip,
            'losses_V': losses_V,
            'losses_entropy': losses_entropy,
            'stds': stds,
            'times_sample': times_sample,
            'times_opt': times_opt,
        }