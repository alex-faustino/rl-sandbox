# -*- coding: utf-8 -*-
"""
Created on Thu Sep 4 01:58:53 2018

@author: sreen
"""

import logging
import math

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import random
import pygame, sys
from pygame.locals import *

#colors
WHITE             = (255,255,255)
RED               = (255,0,0)
GREEN             = (0,255,0)
BLACK             = (0,0,0)

#globals
MAX_BALL_VEL      = 20
WIDTH             = 600
HEIGHT            = 400
BALL_RADIUS       = 20
PAD_WIDTH         = 8
PAD_HEIGHT        = 80
HALF_PAD_WIDTH    = PAD_WIDTH // 2
HALF_PAD_HEIGHT   = PAD_HEIGHT // 2

paddle1_pos       = [HALF_PAD_WIDTH - 1,HEIGHT//2]
paddle2_pos       = [WIDTH +1 - HALF_PAD_WIDTH,HEIGHT//2]
paddle1_vel       = 0
paddle2_vel       = 0
ball_pos          = [WIDTH//2, HEIGHT//2]
ball_vel          = [0, 0]

r_score_threshold = 100
l_score_threshold = 100

max_paddle1_vel   = 20
max_paddle2_vel   = 20
min_paddle1_vel   = -20
min_paddle2_vel   = -20

max_paddle1_pos  = HEIGHT - HALF_PAD_HEIGHT
max_paddle2_pos  = HEIGHT - HALF_PAD_HEIGHT
max_ball_pos     = [WIDTH - BALL_RADIUS, HEIGHT - BALL_RADIUS]
max_ball_vel     = [MAX_BALL_VEL, MAX_BALL_VEL]
min_paddle1_pos  = HALF_PAD_HEIGHT
min_paddle2_pos  = HALF_PAD_HEIGHT
min_ball_pos     = [BALL_RADIUS, BALL_RADIUS]
min_ball_vel     = [-MAX_BALL_VEL, -MAX_BALL_VEL]

logger = logging.getLogger(__name__)
global l_score, r_score, reward, reward_curr
global paddle1_pos, paddle2_pos, ball_pos, ball_vel

class PongEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        global l_score, r_score, reward, reward_curr
        global high, low
        global paddle1_pos, paddle2_pos, ball_pos, ball_vel, paddle1_vel, paddle2_vel
        global ball_pos, ball_vel
        def ball_init():
            global ball_vel
            horz     = 0
            vert     = 0
            while (horz == 0) or (vert == 0):
                horz     = random.randrange(-MAX_BALL_VEL,MAX_BALL_VEL)
                vert     = random.randrange(-MAX_BALL_VEL,MAX_BALL_VEL)
        
            if random.randrange(0,2) is not 0:
                horz = - horz

            ball_vel = [horz, -vert]
    
        r_score = 0
        l_score = 0
        reward  = 0
        
        high = np.array([max_paddle1_pos, max_paddle1_vel, max_paddle2_pos, max_paddle2_vel, max_ball_pos[0], max_ball_pos[1], max_ball_vel[0], max_ball_vel[1]])
        low = np.array([min_paddle1_pos, min_paddle1_vel, min_paddle2_pos, min_paddle2_vel, min_ball_pos[0], min_ball_pos[1], min_ball_vel[0], min_ball_vel[1]])
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high)
        
        self._seed()
        self.viewer = None
        self.state = None
        
        self.steps_beyond_done = None
        self.steps_to_done = 0

        self.state = self.np_random.uniform(low, high, size=(8,))
        state = self.state
        (paddle1_pos[1], paddle1_vel, paddle2_pos[1], paddle2_vel, ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1]) = state

        ball_pos = [WIDTH//2, HEIGHT//2]
        ball_init()

        self.state = (paddle1_pos[1], paddle1_vel, paddle2_pos[1], paddle2_vel, ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps_to_done += 1;
        global paddle1_pos, paddle2_pos, paddle1_vel, paddle2_vel, l_score, r_score
        global l_score, r_score, reward, reward_curr
        global ball_pos, ball_vel
        def ball_init():
            global ball_vel
            horz     = 0
            vert     = 0
            while (horz == 0) or (vert == 0):
                horz     = random.randrange(-MAX_BALL_VEL,MAX_BALL_VEL)
                vert     = random.randrange(-MAX_BALL_VEL,MAX_BALL_VEL)
            
            if random.randrange(0,2) is not 0:
                horz = - horz
            
            ball_vel = [horz, -vert]
        
        assert self.action_space.contains(action), "%r (%s) invalid" %(action, type(action))
        state = self.state
        (paddle1_pos[1], paddle1_vel, paddle2_pos[1], paddle2_vel, ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1]) = state
        
        #update paddle velocity
        if ball_pos[1] < paddle1_pos[1]:
            paddle1_vel = -max_paddle2_vel
        elif ball_pos[1] > paddle1_pos[1]:
            paddle1_vel = max_paddle2_vel
        else:
            paddle1_vel = 0
        
        if action == 0:
            paddle2_vel = -max_paddle1_vel
        elif action == 1:
            paddle2_vel = max_paddle1_vel
        elif action == 2:
            paddle2_vel = 0

        # update paddle's vertical position, keep paddle on the screen
        if paddle1_pos[1] > HALF_PAD_HEIGHT and paddle1_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
            paddle1_pos[1] += paddle1_vel
        elif paddle1_pos[1] < HALF_PAD_HEIGHT and paddle1_vel > 0:
            paddle1_pos[1] += paddle1_vel
        elif paddle1_pos[1] > HEIGHT - HALF_PAD_HEIGHT and paddle1_vel < 0:
            paddle1_pos[1] += paddle1_vel
        elif paddle1_pos[1] < HALF_PAD_HEIGHT and paddle1_vel < 0:
            paddle1_pos[1] -= paddle1_vel
        elif paddle1_pos[1] > HEIGHT - HALF_PAD_HEIGHT and paddle1_vel > 0:
            paddle1_pos[1] -= paddle1_vel
    
        if paddle2_pos[1] > HALF_PAD_HEIGHT and paddle2_pos[1] < HEIGHT - HALF_PAD_HEIGHT:
            paddle2_pos[1] += paddle2_vel
        elif paddle2_pos[1] < HALF_PAD_HEIGHT and paddle2_vel > 0:
            paddle2_pos[1] += paddle2_vel
        elif paddle2_pos[1] > HEIGHT - HALF_PAD_HEIGHT and paddle2_vel < 0:
            paddle2_pos[1] += paddle2_vel
        elif paddle2_pos[1] < HALF_PAD_HEIGHT and paddle2_vel < 0:
            paddle2_pos[1] -= paddle2_vel
        elif paddle2_pos[1] > HEIGHT - HALF_PAD_HEIGHT and paddle2_vel > 0:
            paddle2_pos[1] -= paddle2_vel

        # update ball
        ball_pos[0] += int(ball_vel[0])
        ball_pos[1] += int(ball_vel[1])
        
        # ball collision check on top and bottom walls
        if int(ball_pos[1]) <= BALL_RADIUS:
            ball_vel[1] = - ball_vel[1]
        if int(ball_pos[1]) >= HEIGHT + 1 - BALL_RADIUS:
            ball_vel[1] = - ball_vel[1]
        
        # ball collison check on gutters or paddles
        if int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH and int(ball_pos[1]) in range(int(paddle1_pos[1] - HALF_PAD_HEIGHT), int(paddle1_pos[1] + HALF_PAD_HEIGHT), 1):
            #reward -= 1
            ball_vel[0]  = -ball_vel[0]
            ball_vel[0]  *= 1.1
            ball_vel[1]  *= 1.1
        elif int(ball_pos[0]) <= BALL_RADIUS + PAD_WIDTH:
            #reward      += 2
            r_score      += 1
            ball_pos     = [WIDTH//2, HEIGHT//2]
            ball_init()

        if int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH and int(ball_pos[1]) in range(int(paddle2_pos[1] - HALF_PAD_HEIGHT), int(paddle2_pos[1] + HALF_PAD_HEIGHT), 1):
            reward += 1
            ball_vel[0]  = -ball_vel[0]
            ball_vel[0] *= 1.1
            ball_vel[1] *= 1.1
        elif int(ball_pos[0]) >= WIDTH + 1 - BALL_RADIUS - PAD_WIDTH:
            #reward     -= 2
            l_score     += 1
            ball_pos = [WIDTH//2, HEIGHT//2]
            ball_init()

        #reward = reward
        self.state = (paddle1_pos[1], paddle1_vel, paddle2_pos[1], paddle2_vel, ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1])

        done =  (l_score > l_score_threshold) or (r_score > r_score_threshold)
        done = bool(done)
        
        done = bool(done)
        
        if not done:
            self.steps_beyond_done = self.steps_beyond_done
            reward = reward
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = reward
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            self.steps_to_done      = 0
            reward = 0

        if r_score > r_score_threshold:
            r_score = 0
            l_score = 0
            reward_curr = 0

        if l_score > l_score_threshold:
            r_score = 0
            l_score = 0
            reward_curr = 0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.steps_to_done = 0
        self.steps_beyond_done = None
        global l_score, r_score, reward, reward_curr
        global paddle1_pos, paddle2_pos, ball_pos, ball_vel
        def ball_init():
            global ball_vel
            horz     = 0
            vert     = 0
            while (horz == 0) or (vert == 0):
                horz     = random.randrange(-MAX_BALL_VEL,MAX_BALL_VEL)
                vert     = random.randrange(-MAX_BALL_VEL,MAX_BALL_VEL)
            
            if random.randrange(0,2) is not 0:
                horz = - horz
            
            ball_vel = [horz, -vert]
        
        r_score = 0
        l_score = 0
        reward  = 0
        
        self.state = self.np_random.uniform(low, high, size=(8,))
        state = self.state
        (paddle1_pos[1], paddle1_vel, paddle2_pos[1], paddle2_vel, ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1]) = state
        
        ball_pos = [WIDTH//2, HEIGHT//2]
        ball_init()
        
        self.state = (paddle1_pos[1], paddle1_vel, paddle2_pos[1], paddle2_vel, ball_pos[0], ball_pos[1], ball_vel[0], ball_vel[1])

        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                pygame.quit()
                sys.exit()
            return
    
        if self.viewer is None:
            pygame.init()
            fps = pygame.time.Clock()
            
            #canvas declaration
            window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
            pygame.display.set_caption('Hello World')
            
            window.fill(BLACK)
            pygame.draw.line(window, WHITE, [WIDTH // 2, 0],[WIDTH // 2, HEIGHT], 1)
            pygame.draw.line(window, WHITE, [PAD_WIDTH, 0],[PAD_WIDTH, HEIGHT], 1)
            pygame.draw.line(window, WHITE, [WIDTH - PAD_WIDTH, 0],[WIDTH - PAD_WIDTH, HEIGHT], 1)
            pygame.draw.circle(window, WHITE, [WIDTH//2, HEIGHT//2], 70, 1)
            
            # draw paddles and ball
            ball_pos_int = [int(ball_pos[0]), int(ball_pos[1])]
            pygame.draw.circle(window, RED, ball_pos_int, 20, 0)
            pygame.draw.polygon(window, GREEN, [[paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT], [paddle1_pos[0] - HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] + HALF_PAD_HEIGHT], [paddle1_pos[0] + HALF_PAD_WIDTH, paddle1_pos[1] - HALF_PAD_HEIGHT]], 0)
            pygame.draw.polygon(window, GREEN, [[paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT], [paddle2_pos[0] - HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] + HALF_PAD_HEIGHT], [paddle2_pos[0] + HALF_PAD_WIDTH, paddle2_pos[1] - HALF_PAD_HEIGHT]], 0)
            
            #update scores
            myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
            label1 = myfont1.render("Score: "+str(l_score), 1, (255,255,0))
            window.blit(label1, (50,20))
            
            myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
            label2 = myfont2.render("Reward: "+str(reward), 1, (255,255,0))
            window.blit(label2, (210, 20))
            
            myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
            label2 = myfont2.render("Score: "+str(r_score), 1, (255,255,0))
            window.blit(label2, (470, 20))
            
            pygame.display.update()
            fps.tick(200)
                                                        
        if self.state is None: return None
        
        else: return None


env = PongEnv()
reward = 0

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print('Observation : [%s], Reward: %d'%(','.join(map(str,observation)),reward))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
