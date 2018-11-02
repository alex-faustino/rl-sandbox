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

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.H+50, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H+50, self.H, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(self.H, self.D_out),
)

  self.loss_fn = torch.nn.MSELoss()

  self.learning_rate = 1e-1
  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
  self.y_pred = self.model(self.x)#prediction step is called forward pass
#  print(self.y_pred)
  self.y_pred2 = self.model(self.x.select(0,0))
  pass

 def predict(self,state):
  self.x = torch.tensor(np.transpose(state[np.newaxis])/self.normalizing_states,requires_grad=True).float()
  self.y_pred2 = self.model(self.x.select(0,0))#prediction step is called forward pass
  if state[0] == 15:#shows whether NN is learning
   print('predict')
   print(self.y_pred2)
   print(self.loss.item()) # to see training
  return self.y_pred2

 def reportMinibatchSize(self):
  return self.N

 def update(self,state,reward,discount,previous_state,action): 
  self.x = torch.tensor(np.transpose(state[np.newaxis])/self.normalizing_states,requires_grad=True).float()
  y_pred = self.nn2.predict(self.x)
  temp_previous_state = torch.tensor(np.transpose(previous_state[np.newaxis])/self.normalizing_states,requires_grad=True).float()
  self.y_pred = self.model(temp_previous_state)
  [y_pred,argmax_indices] = torch.max(y_pred, 2)
  self.loss = self.loss_fn(self.y_pred[0,action-1],torch.tensor(reward+discount,requires_grad=True).float()*y_pred)#second term

#  print(self.loss.item()) # to see training

  self.optimizer.zero_grad()

  self.loss.backward()#gradient of loss step is called backward pass

  self.optimizer.step()
#  print('model parameters')
#  print(list(self.model.parameters())[0].grad)
  pass
 def transmitModel(self):# output model parameters
  return self.model
