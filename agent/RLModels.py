import random
from itertools import count

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

###The NN model
class DQNControl(nn.Module):
    def __init__(self, agentSize, phaseSize, stepSize,inputDim, outputDim):
        super(DQNControl, self).__init__()
        #self.embed0 = nn.Embedding(agentSize+1, 10)
        self.embed1 = nn.Embedding(phaseSize+1, 2)

        self.fc2 = nn.Linear(inputDim+1, 64)
        #self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64+1+inputDim, outputDim)
        self.fc4 = nn.Linear(64+1+inputDim, outputDim)

    def forward(self, x):
        #embedPart0 = self.embed0(x[:,0].long())
        embedPart1 = self.embed1(x[:,1].long())
        numericPart1 = x[:,2:180]
        x1 = torch.cat((embedPart1, numericPart1), dim=1)

        x2 = self.fc2(x1)

        total = torch.cat([x1, x2], dim=1)
        Q = self.relu(self.fc3(total))
        Q2 = self.relu(self.fc4(total))
        return -1.0*Q, -1.0*Q2


##The NN model
class DQNControl2(nn.Module):
    def __init__(self, agentSize, phaseSize, stepSize,inputDim, outputDim):
        super(DQNControl2, self).__init__()
        self.embed1 = nn.Embedding(phaseSize+1, 2)

        self.fc2 = nn.Linear(inputDim+1, 64)
        #self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64+1+inputDim, outputDim)
        self.fc4 = nn.Linear(64+1+inputDim, outputDim)

    def forward(self, x):
        #embedPart0 = self.embed0(x[:,0].long())
        embedPart1 = self.embed1(x[:,1].long())
        numericPart1 = x[:,2:228]
        x1 = torch.cat((embedPart1, numericPart1), dim=1)

        x2 = self.fc2(x1)

        total = torch.cat([x1, x2], dim=1)
        Q = (self.fc3(total))
        Q2 = (self.fc4(total))
        return 1.0*Q, 1.0*Q2
