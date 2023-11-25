# Convert the DataFrame to a PyTorch tensor

import torch
from torch.utils.data import Dataset

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)

from parameter_sampling import data_lhs

class FireSpreadDataset(Dataset):
    def __init__(self, dataframe):
        # Convert the dataframe to a PyTorch tensor
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming the last column is the target variable (spread rate)
        features = self.data[idx, :-1]
        target = self.data[idx, -1]
        return features, target

# Convert the DataFrame to a PyTorch Dataset
fire_spread_data = FireSpreadDataset(data_lhs)

# Example: accessing the first item in the dataset
first_sample_features, first_sample_target = fire_spread_data[0]
print(first_sample_features, first_sample_target)
