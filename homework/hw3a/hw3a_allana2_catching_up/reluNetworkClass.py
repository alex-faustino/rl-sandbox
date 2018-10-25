# -*- coding: utf-8 -*-
import torch
import numpy as np

class qLearningNetwork(object):
 def __init__(self,env,nn2):
  self.nn2 = nn2
  self.env = env
  self.normalizing_states, self.allowed_actions = env.states_and_actions()
  self.states = np.array(self.normalizing_states)[np.newaxis]
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  self.N, self.D_in, self.H, self.H2, self.D_out = 20, self.states.shape[0], 90, 65, self.allowed_actions.shape[1]
  self.storage = np.array(np.zeros(self.allowed_actions.shape[1]))[np.newaxis]
  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output
#  self.z = torch.randn(self.N, self.D_out)# randomly initialized adjacent input

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
#  self.z = torch.Tensor(next_state)
#  self.storage = np.vstack((self.storage,self.y_pred.detach().numpy()))
  return self.y_pred.detach().numpy()   
 def update(self,reward,previous_q_function,next_q_function,discount,C): 
  y_pred = self.nn2.predict(self.x)
#  self.loss = self.loss_fn(torch.tensor(self.y_pred,requires_grad=True),torch.tensor(reward+discount*previous_q_function,requires_grad=True))#loss calculation for feedback # print(t, loss.item()) # to see training
  self.loss = self.loss_fn(torch.tensor(y_pred,requires_grad=True),torch.tensor(reward+discount*previous_q_function,requires_grad=True))#loss calculation for feedback # print(t, loss.item()) # to see training

  self.optimizer.zero_grad()

  self.loss.backward()#gradient of loss step is called backward pass

  self.optimizer.step()
#  with torch.no_grad():# Update the weights using gradient descent.
#        for param in self.model.parameters():
#             param -= self.learning_rate * param.grad
  pass
 def transmitModel(self):# verified this in script below PyTorch test in Jupyter Notebook
  return self.model
 def printingPred(self):
  return self.storage
