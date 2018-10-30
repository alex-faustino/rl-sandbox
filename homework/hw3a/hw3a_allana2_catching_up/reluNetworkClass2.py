# -*- coding: utf-8 -*-
import torch
import numpy as np

class qLearningNetwork(object):
 def __init__(self,env):
  self.env = env
  self.normalizing_states, self.allowed_actions = env.states_and_actions()
  self.states = np.array(self.normalizing_states)[np.newaxis]
  # N is minibatch size; D_in is input dimension; H is hidden dimension; D_out is output dimension.
  self.N, self.D_in, self.H, self.D_out = 20, self.states.shape[0], 900, self.allowed_actions.shape[1]
  self.C = 50 # update frequency
  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.H+50, bias=True),
    torch.nn.ReLU(),
#    torch.nn.Linear(self.H+100, self.H+50, bias=True),
#    torch.nn.ReLU(),
    torch.nn.Linear(self.H+50, self.H, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.D_out),
)

  self.loss_fn = torch.nn.MSELoss()

  self.y_pred = self.model(self.x)#initial prediction step for randomly initialized weights
  pass
 def predict(self,state):
  temp_state = torch.from_numpy(np.array(state)[np.newaxis]).float()/self.normalizing_states
  self.x = torch.Tensor(temp_state)
  self.y_pred = self.model(self.x)#prediction step is called forward pass
  return self.y_pred.detach().numpy()   
 def transmitMinibatch(self):
  return self.N
 def transmitHiddenDimension(self):
  return self.H
 def update(self,model,C):
  if np.mod(C,self.C) == 0:# update every C samples
   self.model = model
  pass
