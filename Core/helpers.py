import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from Core.torchhelpers.utils import collate_fn

# helper / utility functions

def to_numpy(tensor):
    '''
    Converts an image tensor to numpy. Mainly for plotting purposes
    '''
    if torch.is_tensor(tensor):
        return tensor.permute(1,2,0).detach().cpu().numpy()
    else:
        print('could not convert to np array. Check type of input')
        return tensor

def to_torch(nparray):
    '''
    Converts a np.ndarray to a torch tensor
    '''
    if isinstance(nparray,np.ndarray):
        return torch.from_numpy(nparray)
    else:
        print('could not convert to torch tensor. Check type of input')
        return nparray

def split_data_train_test(DatasetClass,validation_split=0.75,batch_size=1, shuffle_dataset=False, shuffle_seed=None,data_amount=1):
    '''
    Function that splits data from dataset class into a train and validation set

    validation_split: the percentage of the data that should be in the validation set
    '''

    # Creating data indices for training and validation splits:
    dataset_size = int(np.floor(len(DatasetClass)*data_amount))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(DatasetClass, batch_size=batch_size, 
                                            sampler=train_sampler,
                                            collate_fn=collate_fn)
    validation_loader = DataLoader(DatasetClass, batch_size=batch_size,
                                                    sampler=val_sampler,
                                                    collate_fn=collate_fn)
    print(f'Total size of dataset: {dataset_size}\nSize of train data: {len(train_loader.sampler)}\nSize of validation data: {len(validation_loader.sampler)}')
    return train_loader,validation_loader
