#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 03:41:08 2018

@author: vedant
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import numpy as np

m = nn.Linear(6, 1)
state = np.array([1.,2.,3.,4.,5.,6.])
state
ts = (torch.from_numpy(state))
ts
m(ts)
torch.set_default_tensor_type('torch.DoubleTensor')
m(ts)
state

ts = (torch.from_numpy(state))
print(m(ts))
print(m.bias)
print(m.weight)