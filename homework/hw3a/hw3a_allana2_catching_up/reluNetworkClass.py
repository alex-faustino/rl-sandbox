# -*- coding: utf-8 -*-
import torch
import numpy as np

class qLearningNetwork(object):
 def __init__(self,env):
  self.env = env
#  self.agent = agent # I don't think I need this, but am leaving it here for now
  self.states, self.allowed_actions = env.states_and_actions()
  self.states = np.array(self.states)[np.newaxis]
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  if self.states.shape[0] == 1:
   self.N, self.D_in, self.H, self.H2, self.D_out = 1, 1, 90, 65, self.allowed_actions.shape[0]
  else:
   self.N, self.D_in, self.H, self.H2, self.D_out = 1, self.states.shape[1], 90, 65, self.allowed_actions.shape[0]
  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.D_out),
)
  self.loss_fn = torch.nn.MSELoss()

  self.learning_rate = 1e-4
  self.y_pred = self.model(self.x)#initial prediction step is called forward pass
  pass
 def predict(self,state):
#  print(self.x)
#  print(state)
  temp_state = torch.from_numpy(np.array(state*1.0)[np.newaxis]).float()
  self.x = torch.Tensor(temp_state) #so for some reason the dimension of weight m1
  self.y_pred = self.model(self.x)#prediction step is called forward pass
#  print(self.y_pred)
  return self.y_pred.detach().numpy()   
 def update(self,state,reward,previous_q_function): 
  self.loss = self.loss_fn(torch.tensor(self.y_pred,requires_grad=True),torch.tensor(previous_q_function,requires_grad=True))#loss calculation for feedback # print(t, loss.item()) # to see training

  self.loss.backward()#gradient of loss step is called backward pass

  with torch.no_grad():# Update the weights using gradient descent.
        for param in self.model.parameters():
            param -= self.learning_rate * param.grad
  pass
