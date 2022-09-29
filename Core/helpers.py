import torch
import numpy as np

# helper / utility functions

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.permute(1,2,0).detach().cpu().numpy()
    else:
        print('could not convert to np array. Check type of input')
        return tensor

def to_torch(nparray):
    if isinstance(nparray,np.ndarray):
        return torch.from_numpy(nparray)
    else:
        print('could not convert to torch tensor. Check type of input')
        return nparray