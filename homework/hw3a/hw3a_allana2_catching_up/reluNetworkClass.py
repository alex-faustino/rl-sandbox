# -*- coding: utf-8 -*-
import torch
import numpy as np

class qLearningNetwork(object):
 def __init__(self,env):
  self.storage = np.array(0)[np.newaxis]
  self.env = env
  self.states, self.allowed_actions = env.states_and_actions()
  self.states = np.array(self.states)[np.newaxis]
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
#  if self.states.shape[0] == 1:
   self.N, self.D_in, self.H, self.H2, self.D_out = 20, 1, 90, 65, self.allowed_actions.shape[1]
#  else:
#   self.N, self.D_in, self.H, self.H2, self.D_out = 1, self.states.shape[1], 90, 65, self.allowed_actions.shape[1]
  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.D_out),
)
  self.loss_fn = torch.nn.MSELoss()

  self.learning_rate = 1e-4
  self.y_pred = self.model(self.x)#initial prediction step is called forward pass
  pass
 def predict(self,state):
  temp_state = torch.from_numpy(np.array(state)[np.newaxis]).float()
  self.x = torch.Tensor(temp_state) #so for some reason the dimension of weight m1
  self.y_pred = self.model(self.x)#prediction step is called forward pass
  return self.y_pred.detach().numpy()   
 def update(self,reward,previous_q_function,next_q_function,discount): 
  self.loss = self.loss_fn(torch.tensor(self.y_pred,requires_grad=True),torch.tensor(reward+discount*previous_q_function,requires_grad=True))#loss calculation for feedback # print(t, loss.item()) # to see training

  self.loss.backward()#gradient of loss step is called backward pass

  with torch.no_grad():# Update the weights using gradient descent.
        for param in self.model.parameters():
             param -= self.learning_rate * param.grad
  pass
 def printingPred(self):
  return self.storage