import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
import numpy as np

class EegDataset(Dataset):
    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx:idx+1,:,:], self.labels[idx:idx+1]
