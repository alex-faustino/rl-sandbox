# -*- coding: utf-8 -*-
import torch

class qLearningNetwork(object):
 def __init__(self,env,agent):
  self.env = env
  self.agent = agent # I don't think I need this, but am leaving it here for now
  self.states, self.allowed_actions = env.states_and_actions()
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  self.N, self.D_in, self.H, self.H2, self.D_out = 1, self.states, 100, 75, self.allowed_actions.shape[1]

  self.x = torch.randn(self.N, self.D_in)# randomly initialized input
  self.y = torch.randn(self.N, self.D_out)# randomly initialized output

  self.model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.RelU(),
    torch.nn.Linear(self.H, self.D_out),
)
  self.loss_fn = torch.nn.MSELoss()

  self.learning_rate = 1e-4
  pass
 def predict(self,state):
  self.y_pred = self.model(state)#prediction step is called forward pass
  return self.y_pred.detach().numpy()   
 def update(self,state,reward,previous_q_function): 
  self.loss = loss_fn(y_pred, y)#loss calculation for feedback # print(t, loss.item()) # to see training

  self.loss.backward()#gradient of loss step is called backward pass

    with torch.no_grad():# Update the weights using gradient descent.
        for param in model.parameters():
            param -= self.learning_rate * param.grad
  pass
