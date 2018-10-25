# -*- coding: utf-8 -*-
import torch
import numpy as np

class qLearningNetwork(object):
 def __init__(self,env):
  self.env = env
  self.normalizing_states, self.allowed_actions = env.states_and_actions()
  self.states = np.array(self.normalizing_states)[np.newaxis]
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  self.N, self.D_in, self.H, self.H2, self.D_out = 20, self.states.shape[0], 90, 65, self.allowed_actions.shape[1]
  self.storage = np.array(np.zeros(self.allowed_actions.shape[1]))[np.newaxis]
  self.C = 625
  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.D_out),
)

  self.loss_fn = torch.nn.MSELoss()

  self.learning_rate = 1e-4
  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
  self.y_pred = self.model(self.x)#initial prediction step is called forward pass
  self.storage = np.vstack((self.storage,self.y_pred.detach().numpy()))
  pass
 def predict(self,state):
  temp_state = torch.from_numpy(np.array(state)[np.newaxis]).float()/self.normalizing_states
  self.x = torch.Tensor(temp_state) #so for some reason the dimension of weight m1
  self.y_pred = self.model(self.x)#prediction step is called forward pass
#  self.storage = np.vstack((self.storage,self.y_pred.detach().numpy()))
  return self.y_pred.detach().numpy()   
 def update(self,model,C):
  if np.mod(C,self.C) == 0:# update every C samples
#   print(C)
   self.model = model
  pass
 def printingPred(self):
  return self.storage
