import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VideoData(Dataset):
    def __init__(self, samples, t):
        self.samples = samples
        self.T = t

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return self.T(sample), target

    def __len__(self):
        return len(self.samples)


def load_data(batch_size=64):
    train_set = VideoData('data/train', transforms.ToTensor())
    test_set = VideoData('data/test', transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

