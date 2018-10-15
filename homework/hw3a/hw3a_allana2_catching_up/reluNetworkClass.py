# -*- coding: utf-8 -*-
import torch

class qLearningNetwork(object):
 def __init__(self,env):
 self.env = env
 self.gridnum, self.allowed_actions = env.states_and_actions()
 self.q_function = q_function
 
 # N is batch size; D_in is input dimension;
 # H is hidden dimension; D_out is output dimension.
 self.N, self.D_in, self.H, self.H2, self.D_out = 64, self.gridnum**2, 100, 100, self.allowed_actions.shape[1]

 # Create random Tensors to hold inputs and outputs
 self.x = torch.randn(self.N, self.D_in)
 self.y = torch.randn(self.N, self.D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(self.D_in, self.H),
    torch.nn.ReLU(),
    torch.nn.RelU(),
    torch.nn.Linear(self.H, self.D_out),
)

loss_fn = torch.nn.MSELoss()

learning_rate = 1e-4
for t in range(1000):
    y_pred = model(x)#prediction step is called forward pass

    loss = loss_fn(y_pred, y)#loss calculation for feedback
    print(t, loss.item())

    loss.backward()#gradient of loss step is called backward pass

    with torch.no_grad():# Update the weights using gradient descent.
        for param in model.parameters():
            param -= learning_rate * param.grad
