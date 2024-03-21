"""
Datasets reading functions

Reference: Pending

@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import os
import torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

' Define the reading of the dataset '
class PUDataset(Dataset):
    def __init__(self, ids, dir_input, dir_gt, extension='.mat'):
        self.dir_input = dir_input
        self.dir_gt = dir_gt
        self.extension = extension
        self.ids = ids # Dataset IDS
        self.data_len = len(self.ids) # Calculate len of data

    ' Ask for input and ground truth'
    def __getitem__(self, index):

        # Get an ID of the input and ground truth
        id_input = self.dir_input + self.ids[index] + self.extension
        id_gt = self.dir_gt + self.ids[index] + self.extension
        # Open them
        input = sio.loadmat(id_input)
        gt = sio.loadmat(id_gt)
        input = input['input']
        gt = gt['gt']
        input = torch.from_numpy(input).float().unsqueeze(0)
        gt = torch.from_numpy(gt).float().unsqueeze(0)
        return (input, gt)

    ' Length of the dataset '
    def __len__(self):
        return self.data_len

' Return the training dataset separated in batches '
def get_dataloaders(dir_input, dir_gt, batch_size=10):
    ids = [f[:-4] for f in os.listdir(dir_input)]  # Read the names of the images
    dset = PUDataset(ids, dir_input, dir_gt)  # Get the dataset
    dataloaders = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=False)  # Create the dataloaders

    return dataloaders

' Return the training dataset separated in batches '
def get_dataloaders_th(dir_input, dir_gt, th):
    ids = [f[:-4] for f in sorted(os.listdir(dir_input))][th:th+1]  # Read the names of the images
    dset = PUDataset(ids, dir_input, dir_gt)  # Get the dataset
    dataloaders = DataLoader(dset, batch_size=1, drop_last=False)  # Create the dataloaders
    return dataloaders

' Return the dataset for testing '
def get_dataloader_for_test(dir_img, dir_gt):
    ids = [f[:-4] for f in sorted(os.listdir(dir_img))]  # Read the names of the datas
    dset = PUDataset(ids, dir_img, dir_gt)  # Get the dataset
    dataloader = DataLoader(dset, batch_size=len(ids))  # Create the dataloader
    return dataloader


