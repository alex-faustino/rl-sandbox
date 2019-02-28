import numpy as np

# Pretty standard implementations of SARSA and Q-Learning in Python
# Adapted from: github.com/MorvanZhou/Reinforcement-learning-with-tensorflow


class GenTableMethodsRL(object):
    def __init__(self, env, learning_rate, reward_decay, eps):
        self.env = env
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.eps = eps

        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, s):
        if np.random.rand() < (1 - self.eps):
            # exploitation
            a = self.q_table[s, :].argmax()
        else:
            # exploration
            a = self.env.action_space.sample()
        return a

    def train(self, *args):
        pass

    def learn(self, *args):
        pass


class Sarsa(GenTableMethodsRL):
    def __init__(self, env, learning_rate, reward_decay, eps):
        super(Sarsa, self).__init__(env, learning_rate, reward_decay, eps)

    def train(self, episodes_num, episode_length, viz):
        all_cum_rewards = []

        for episode in range(episodes_num):
            s = self.env.reset()
            a = self.choose_action(s)
            cum_reward = 0
            for t in range(episode_length):
                if viz:
                    self.env.render()

                s_plus1, r, done, info = self.env.step(a)
                a_plus1 = self.choose_action(s_plus1)
                if t == episode_length - 1:  # temporary way to end each episode cleanly
                    s_plus1 = 'terminal'

                # update q
                self.learn(s, a, r, s_plus1, a_plus1)

                # cycle
                s = s_plus1
                a = a_plus1

                # accumulate reward
                cum_reward += r

            all_cum_rewards.append(cum_reward)

        return all_cum_rewards

    def learn(self, s, a, r, s_plus1, a_plus1):
        q_predict = self.q_table[s, a]
        if s_plus1 != 'terminal':
            q_target = r + self.reward_decay*self.q_table[s_plus1, a_plus1]
        else:
            q_target = r
        self.q_table[s, a] += self.learning_rate*(q_target - q_predict)


class QLearning(GenTableMethodsRL):
    def __init__(self, env, learning_rate, reward_decay, eps):
        super(QLearning, self).__init__(env, learning_rate, reward_decay, eps)

    def train(self, episodes_num, episode_length, viz):
        all_cum_rewards = []

        for episode in range(episodes_num):
            s = self.env.reset()
            cum_reward = 0
            for t in range(episode_length):
                if viz:
                    self.env.render()

                a = self.choose_action(s)
                s_plus1, r, done, info = self.env.step(a)
                if t == episode_length - 1:  # temporary way to end each episode cleanly
                    s_plus1 = 'terminal'

                # update q
                self.learn(s, a, r, s_plus1)

                # cycle
                s = s_plus1

                # accumulate reward
                cum_reward += r
            all_cum_rewards.append(cum_reward)

        return all_cum_rewards

    def learn(self, s, a, r, s_plus1):
        q_predict = self.q_table[s, a]
        if s_plus1 != 'terminal':
            q_target = r + self.reward_decay*self.q_table[s_plus1, :].max()
        else:
            q_target = r
        self.q_table[s, a] += self.learning_rate * (q_target - q_predict)


class Reinforce(GenTableMethodsRL):
    def __init__(self, env, learning_rate, reward_decay, eps):
        super(Reinforce, self).__init__(env, learning_rate, reward_decay, eps)

    def sample_trajectory(self, episode_length, policy):
        # get states, actions, and rewards for trajectory
        traj_s, traj_a, traj_r = np.zeros([episode_length], dtype=int),\
                                 np.zeros([episode_length], dtype=int),\
                                 np.zeros([episode_length], dtype=int)

        traj_s[0] = self.env.reset()
        for t in range(episode_length):
            traj_a[t] = reinforce_choose_action(traj_s[t], policy)
            s, traj_r[t], _, _ = self.env.step(traj_a[t])

            if t != episode_length - 1:
                traj_s[t + 1] = s

        return traj_s, traj_a, traj_r

    def train(self, episodes_num, batch_size, episode_length):
        batches_num = int(episodes_num/batch_size)
        policy = np.ones((self.env.observation_space.n, self.env.action_space.n))
        all_cum_rewards = []

        for batch in range(batches_num):

            grad_J = np.zeros(policy.shape)
            reward_avg = 0
            for n in range(batch_size):

                s, a, r = self.sample_trajectory(episode_length, policy)

                J = np.zeros(policy.shape)
                reward = 0
                for t in range(episode_length):
                    # update the gradient
                    J += est_grad(s[t], a[t], policy)
                    reward += r[t]

                grad_J += (J*reward)/batch_size

                reward_avg += reward / batch_size

            # learn
            all_cum_rewards.append(reward_avg)
            policy = policy + self.learning_rate * grad_J

        return np.arange(batches_num)*batch_size, all_cum_rewards, policy


def reinforce_choose_action(s, policy):
    # softmax
    policy_a = np.exp(policy[s, :]) / np.sum(np.exp(policy[s, :]), axis=0)
    a = np.random.choice(np.arange(policy.shape[1]), 1, replace=False, p=policy_a)

    return int(a)


def est_grad(s, a, policy):
    grad_J = np.zeros(policy.shape)
    policy_a = np.exp(policy[s, :]) / np.sum(np.exp(policy[s, :]), axis=0)

    grad_J[s, :] = -policy_a[a]
    grad_J[s, a] = 1 - policy_a[a]

    return grad_J
