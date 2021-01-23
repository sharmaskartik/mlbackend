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


def generate_partitions(num_samples, nfolds, fold_id):

    fractions = 1 / nfolds
    partition_idxs = [int(i * fractions * num_samples) for i in range(1, nfolds)]

    idxs = np.arange(0, num_samples)
    np.random.shuffle(idxs)
    splits = np.split(idxs, partition_idxs)
    valid_partition_id = int(fold_id / (nfolds-1))
    test_partition_id = int(fold_id % (nfolds-1))
    if test_partition_id >= valid_partition_id:
        test_partition_id +=1

    assert test_partition_id != valid_partition_id, 'ERROR! Same id assigned to validation and test partition'
    valid_idxs = np.array(splits[valid_partition_id]).  reshape(-1)
    test_idxs = np.array(splits[test_partition_id]).reshape(-1)
    train_split_ids = np.delete(np.arange(5), [valid_partition_id, test_partition_id]).astype(int)
    train_idxs = np.array(splits)[train_split_ids].reshape(-1)

    return train_idxs, valid_idxs, test_idxs

