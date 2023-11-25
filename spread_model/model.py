import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List 
import torch.nn as nn
import torch.optim as optim

class BushfireModel(nn.Module): 
    def __init__(self, n: int, hidden_layers: List[int]): 
        super().__init__() #Inherit from the nn.module
        self.n = n # n input features
        self.hidden_layers = hidden_layers # hidden layers are the middle ones

        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([n, *hidden_layers], [*hidden_layers, 1])
        ]) # Single output

    def forward(self, x: torch.tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:    #We don't want to ReLU the last layer
                x = nn.functional.relu(x)  
        return x # Linear output for regression