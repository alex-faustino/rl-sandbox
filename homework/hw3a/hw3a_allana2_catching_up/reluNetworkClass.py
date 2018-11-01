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

  self.learning_rate = 1e-3
  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
  self.y_pred = self.model(self.x)#prediction step is called forward pass
  print(self.y_pred)
  self.y_pred2 = self.model(self.x.select(0,2))
  pass
 def predict(self,state):
  temp_state = torch.from_numpy(np.transpose(np.array(state)[np.newaxis])).float()/self.normalizing_states
  self.x = torch.Tensor(temp_state)
  self.y_pred = self.model(self.x.select(0,0))#prediction step is called forward pass
  if state[0] == 15:#shows whether NN is learning
   print('predict')
   print(self.y_pred)
  self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
  return self.y_pred.detach().numpy()
 def reportMinibatchSize(self):
  return self.N
 def update(self,state,reward,discount,previous_state,action): 
  temp_state = torch.from_numpy(np.array(state)[np.newaxis]).float()/self.normalizing_states
  temp_previous_state = torch.from_numpy(np.transpose(np.array(previous_state)[np.newaxis])).float()/self.normalizing_states
  temp_state = temp_state[0]
  self.x = torch.Tensor(temp_state)
  y_pred = self.nn2.predict(self.x)
  y_pred = y_pred[0]
  self.y_pred = self.model(temp_previous_state)
  placeholder_for_predict = np.amax(y_pred,axis=1)*0
  placeholder_for_predict =   placeholder_for_predict.astype(float)
  temp_y_pred = self.y_pred.detach().numpy()
  for z in range(0,self.N):
   placeholder_for_predict[z] = temp_y_pred[z,action[z]-1]
  self.y_pred = torch.from_numpy(placeholder_for_predict)
  self.loss = self.loss_fn(torch.tensor(self.y_pred),torch.tensor(reward+discount*np.amax(y_pred,axis=1),requires_grad=True))#second term

#  print(self.loss.item()) # to see training

  self.optimizer.zero_grad()

  self.loss.backward()#gradient of loss step is called backward pass

  self.optimizer.step()
  pass
 def transmitModel(self):# output model parameters
  return self.model
