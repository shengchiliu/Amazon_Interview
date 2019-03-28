from __future__ import print_function
import os
from os import listdir
from os.path import join

import numpy as np
from scipy.misc import imread, imsave, imrotate, imresize
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, ToPILImage
from torchvision.utils import save_image
import torch.utils.data as data
from torch.utils.data import DataLoader



class DatasetFromNPZ(data.Dataset):

    def __init__(self, path_npz, npz_transform=None):
        super(DatasetFromNPZ, self).__init__()

        # .npz dataset
        dataset = np.load(path_npz)
        self.h_input  = dataset["h0_2ch_nor"]           # [len_dataset, 2, h0]
        _, d, h = self.h_input.shape

        # Numpy Array >> Torch Tensor
        self.h_input  = torch.from_numpy(self.h_input).float().view(-1, d*h)

        # Transform
        self.npz_transform = npz_transform

    def __getitem__(self, index):
        h_input = self.h_input[index]
        # c_target = self.c_target[index]
        if self.npz_transform:
            h_input = self.npz_transform(h_input)
            # c_target = self.npz_transform(c_target)
        return h_input

    def __len__(self):
        return len(self.h_input)



# Test Code
if __name__ == "__main__":
    # depth = 2
    dataset = DatasetFromNPZ('./data/dataset.npz', 0)
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
    for batch in dataloader:
        h_input = batch
    print('h_input  (h): {}\n'.format(h_input.shape))

