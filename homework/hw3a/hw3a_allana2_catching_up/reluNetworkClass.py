# -*- coding: utf-8 -*-
import torch
import numpy as np

class qLearningNetwork(object):
 def __init__(self,env,nn2):
  self.nn2 = nn2
  self.env = env
  self.normalizing_states, self.allowed_actions = env.states_and_actions()
  self.states = np.array(self.normalizing_states)[np.newaxis]
  # N is minibatch size; D_in is input dimension; H is hidden dimension; D_out is output dimension.
  self.N, self.D_in, self.H, self.D_out = self.nn2.transmitMinibatch(), self.states.shape[0], self.nn2.transmitHiddenDimension(), self.allowed_actions.shape[1]
  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.H+100, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H+100, self.H+50, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H+50, self.H, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.D_out),
)

  self.loss_fn = torch.nn.MSELoss()

  self.learning_rate = 1e-3
  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
  self.y_pred = self.model(self.x)#initial prediction step for randomly initialized weights
  pass
 def predict(self,state):
  temp_state = torch.from_numpy(np.array(state)[np.newaxis]).float()/self.normalizing_states
  self.x = torch.Tensor(temp_state).unsqueeze(0)
  self.y_pred = self.model(self.x)#prediction step is called forward pass
  return self.y_pred.detach().numpy()   
 def reportMinibatchSize(self):
  return self.N
 def update(self,state,reward,previous_q_function,discount): 
  temp_state = torch.from_numpy(np.array(state)[np.newaxis]).float()/self.normalizing_states
  self.x = torch.Tensor(temp_state).unsqueeze(0)
  y_pred = self.nn2.predict(self.x)
  y_pred = y_pred[0] # necessary since y_pred is numpy array of size 1 x 1 containing a numpy array of size 1 x 4
  self.loss = self.loss_fn(torch.tensor(previous_q_function,requires_grad=True),torch.tensor(reward+discount*np.amax(y_pred),requires_grad=True))#second term is target and first term is estimate

#  print(self.loss.item()) # to see training

  self.optimizer.zero_grad()

  self.loss.backward()#gradient of loss step is called backward pass

  self.optimizer.step()
  pass
 def transmitModel(self):# output model parameters
  return self.model
