from pprint import pprint
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

import importlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import gym.spaces
import gym
import FastSLAM
import hw6_ppo

credentials = GoogleCredentials.get_application_default()
service = discovery.build('compute', 'v1', credentials=credentials)
project = 'ae598rl'
zone = 'us-central1-c'
instance = 'instance-2'
request = service.instances().stop(project=project, zone=zone, instance=instance)

env = gym.make('FastSLAM-v0')
agent = hw6_ppo.PPOAgent(env)

save_model = 'case3'
load_model = 'case3_LM'
load_vars = False

gamma = 0.99
lamb = 0.95
number_of_actors = 40
number_of_iterations = 100
horizon = 100
number_of_epochs = 100
minibatch_size = 64
logstd_initial = -1 #-0.7
logstd_final = -2 # -1.6
epsilon = 0.2
use_multiprocess = True

res = agent.train(
    save_model,
    load_model,
    load_vars,
    gamma,
    lamb,
    number_of_actors,
    number_of_iterations,
    horizon,
    number_of_epochs,
    minibatch_size,
    logstd_initial,
    logstd_final,
    epsilon,
    use_multiprocess,
)

response = request.execute()
pprint(response)