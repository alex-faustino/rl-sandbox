import numpy as np

import argparse
from copy import deepcopy
import torch
import gym

import matplotlib.pyplot as plt
from ddpg import DDPG
from util import *
from slide import SlideEnv

replay_k = 4

def train(epochs, agent, env, cycles, Ts):
    total_success = []
    average_reward = []
    acc_success = 0
    acc_reward = 0
    for epoch in range(epochs):
        observation = env.reset()
        agent.reset(observation)
        episode_reward = 0
        success = 0
        Episode = []
        count = 0
        for t in range(Ts):
            action = agent.select_action(observation)
            observation2, reward, done, info = env.step(action)
            if t == Ts - 1:
                done = 1
            Episode.append([observation, action, reward, observation2, done])
            agent.observe(observation, action, reward, observation2, done)
            observation = np.copy(observation2)
            count = t + 1
            episode_reward += reward
            if info == 'congratulation!':
                success = 1
            if done == 1:
                break
        for t in range(count):
            idx = np.random.randint(t, count, replay_k)
            for i in idx:
                s0, a, r, s1, done = Episode[t]
                r1 = env.cal_reward(Episode[t][0], Episode[i][0])
                agent.observe(s0, a, r1, s1, done)

        print('#{}: episode_reward:{}, success:{}'.format(epoch,episode_reward,success))
        agent.update_policy()
        acc_success += success
        acc_reward += episode_reward
        if (epoch+1) % cycles == 0:
            total_success.append(acc_success)
            average_reward.append(acc_reward/cycles)
            acc_success = 0
            acc_reward = 0
    x = [i for i in range(len(total_success))]
    plt.xlabel('Cycle')
    plt.ylabel('Average Reward')
    plt.plot(x, average_reward)
    plt.savefig('average_reward.png')
    plt.close()
    plt.xlabel('Cycle')
    plt.ylabel('Success')
    plt.plot(x, total_success)
    plt.savefig('total_success.png')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--hidden1', default=40, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=30, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--Ts', default=20, type=int, help='')
    parser.add_argument('--cycles', default=50, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--epochs', default=2000, type=int, help='train iters each timestep')

    args = parser.parse_args()

    env = SlideEnv()

    nb_states = env.observation_space
    nb_actions = env.action_space

    agent = DDPG(nb_states, nb_actions, args)
    train(args.epochs, agent, env, args.cycles, args.Ts)
